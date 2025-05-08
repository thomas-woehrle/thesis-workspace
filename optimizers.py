import torch
import torch.nn as nn
from torch._functorch.functional_call import functional_call
from torch._functorch.apis import vmap


class OpenAIEvolutionaryOptimizer:
    def __init__(
        self,
        popsize: int,
        sigma: float,
        lr: float,
        model: nn.Module,
        use_antithetic_sampling: bool,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.popsize = popsize
        self.sigma = sigma
        self.lr = lr
        self.use_antithetic_sampling = use_antithetic_sampling
        self.model = model
        self.batched_named_buffers = {
            n: b.repeat(self.popsize, 1) for n, b in self.model.named_buffers()
        }
        self.params_vector = nn.utils.parameters_to_vector(model.parameters())
        self._epsilon = torch.zeros(popsize, len(self.params_vector))
        self.device = device
        self.dtype = dtype

    def prepare_mutations(self):
        """Creates new epsilon. Epsilon is of shape popsize * num_params"""
        if self.use_antithetic_sampling:
            assert self.popsize % 2 == 0, "If using antithetic sampling, the popsize has to be even"

            # This seemingly weird direct assignment to self._epsilon is done to not waste RAM
            self._epsilon = torch.randn(
                self.popsize // 2, len(self.params_vector), device=self.device, dtype=self.dtype
            )
            self._epsilon = torch.concatenate([self._epsilon, -self._epsilon], dim=0)
        else:
            self._epsilon = torch.randn(
                self.popsize, len(self.params_vector), device=self.device, dtype=self.dtype
            )

    def load_mutation_into_model(self, mutation_idx: int):
        candidate_vector = self.params_vector + self._epsilon[mutation_idx] * self.sigma
        nn.utils.vector_to_parameters(candidate_vector, self.model.parameters())

    def parallel_forward_pass(self, x: torch.Tensor):
        batched_flat_params = self.params_vector + self.sigma * self._epsilon
        batched_flat_params_split = batched_flat_params.split(
            [p.numel() for p in self.model.parameters()], dim=1
        )

        batched_named_params = {
            n: batched_flat_p.view(self.popsize, *p.shape)
            for (n, p), batched_flat_p in zip(
                self.model.named_parameters(), batched_flat_params_split
            )
        }

        def forward_pass(named_params, named_buffers):
            return functional_call(self.model, (named_params, named_buffers), (x,))

        batched_forward_pass = vmap(forward_pass, in_dims=(0, 0))
        return batched_forward_pass(batched_named_params, self.batched_named_buffers)

    def step(self, losses: torch.Tensor):
        # losses of shape popsize x 1
        # estimate gradients
        normalized_losses = (losses - losses.mean()) / losses.std()
        g_hat = (self._epsilon.T @ normalized_losses).flatten()
        g_hat = g_hat / (self.popsize * self.sigma)
        self.params_vector -= self.lr * g_hat
        nn.utils.vector_to_parameters(self.params_vector, self.model.parameters())

    def get_current_buffers(self) -> dict[str, torch.Tensor]:
        # return the first of the batched buffers for each
        # does not make a difference, because all of the batched individuals get to see the same input
        return {n: b[0] for n, b in self.batched_named_buffers.items()}


class SimpleEvolutionaryOptimizer:
    def __init__(
        self,
        n_families: int,
        members_per_family: int,
        sigma: float,
        model: nn.Module,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.n_families = n_families
        self.members_per_family = members_per_family
        self.sigma = sigma
        self.model = model
        # every individual starts with the same parameters
        self.family_param_vectors = nn.utils.parameters_to_vector(model.parameters()).repeat(
            n_families, members_per_family, 1
        )
        self.device = device
        self.dtype = dtype

    def mutate(self):
        # parents, ie the first member of each family should not be mutated
        self.family_param_vectors[:, 1:, :] += (
            torch.randn(
                self.family_param_vectors.shape[0],
                self.family_param_vectors.shape[1] - 1,
                self.family_param_vectors.shape[2],
                device=self.device,
                dtype=self.dtype,
            )
            * self.sigma
        )

    def load_individual_into_model(self, family_idx: int, member_idx: int):
        # An individual is identified by its family_idx and member_idx within that family
        individual_param_vector = self.family_param_vectors[family_idx][member_idx]

        nn.utils.vector_to_parameters(individual_param_vector, self.model.parameters())

    def step(self, losses: torch.Tensor):
        # losses of shape n_families x (n_members_per_family)
        normalized_losses = (losses - losses.mean()) / losses.std()

        # get the flat indices of the lowest losses
        _, flat_indices = normalized_losses.flatten().topk(
            self.n_families, largest=False, sorted=True
        )

        # select the parents according to the flat_indices, new_parents of shape n_families x num_params
        new_parents = self.family_param_vectors.flatten(0, 1)[flat_indices,]

        # unsqueezing turns into n_families x 1 x num_params. Then repeat along the 2nd dimension
        self.family_param_vectors = new_parents.unsqueeze(1).repeat(1, self.members_per_family, 1)
