from abc import ABC, abstractmethod
from typing import Iterable
import torch
import torch.nn as nn
import torch.optim as optim


class EvolutionaryOptimizer(ABC, optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float,
        popsize: int,
    ):
        self.params = list(params)
        self.lr = lr
        self.popsize = popsize

    @abstractmethod
    def get_new_generation(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def step(self, losses: torch.Tensor, mutations: torch.Tensor):
        pass


class OpenAIEvolutionaryOptimizer(EvolutionaryOptimizer):
    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float,
        popsize: int,
        sigma: float,
        use_antithetic_sampling: bool,
    ):
        super().__init__(params, lr, popsize)
        self.popsize = popsize
        self.sigma = sigma
        self.use_antithetic_sampling = use_antithetic_sampling
        self.flat_params = nn.utils.parameters_to_vector(self.params)

    def get_new_generation(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.use_antithetic_sampling:
            assert self.popsize % 2 == 0, "If using antithetic sampling, the popsize has to be even"

            mutations = self.sigma * torch.randn(
                self.popsize // 2,
                len(self.flat_params),
                device=self.flat_params.device,
                dtype=self.flat_params.dtype,
            )
            mutations = torch.concatenate([mutations, mutations], dim=0)
        else:
            mutations = self.sigma * torch.randn(
                self.popsize,
                len(self.flat_params),
                device=self.flat_params.device,
                dtype=self.flat_params.dtype,
            )

        # '+' operation broadcasts to (popsize, num_params)
        batched_mutated_flat_params = self.flat_params + mutations

        return batched_mutated_flat_params, mutations

    def step(self, losses: torch.Tensor, mutations: torch.Tensor):
        # losses of shape popsize x 1
        # estimate gradients
        normalized_losses = (losses - losses.mean()) / losses.std()
        g_hat = ((mutations.T / self.sigma) @ normalized_losses).flatten()
        g_hat = g_hat / (self.popsize * self.sigma)
        self.flat_params -= self.lr * g_hat
        nn.utils.vector_to_parameters(self.flat_params, self.params)


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
