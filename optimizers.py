import torch
import torch.nn as nn


class NaturalEvolutionOptimizer:
    def __init__(
        self,
        popsize: int,
        sigma: float,
        lr: float,
        model: nn.Module,
        use_antithetic_sampling: bool,
        device: torch.device,
    ):
        self.popsize = popsize
        self.sigma = sigma
        self.lr = lr
        self.use_antithetic_sampling = use_antithetic_sampling
        self.model = model
        self.params_vector = nn.utils.parameters_to_vector(model.parameters())
        self._epsilon = torch.zeros(popsize, len(self.params_vector))
        self.device = device

    def prepare_mutations(self):
        """Creates new epsilon. Epsilon is of shape popsize * num_params"""
        if self.use_antithetic_sampling:
            assert self.popsize % 2 == 0, "If using antithetic sampling, the popsize has to be even"

            # This seemingly weird direct assignment to self._epsilon is done to not waste RAM
            self._epsilon = torch.randn(
                self.popsize // 2, len(self.params_vector), device=self.device
            )
            self._epsilon = torch.concatenate([self._epsilon, -self._epsilon], dim=0)
        else:
            self._epsilon = torch.randn(self.popsize, len(self.params_vector), device=self.device)

    def load_mutation_into_model(self, mutation_idx: int):
        candidate_vector = self.params_vector + self._epsilon[mutation_idx] * self.sigma
        nn.utils.vector_to_parameters(candidate_vector, self.model.parameters())

    def step(self, losses: torch.Tensor):
        # losses of shape popsize x 1
        # estimate gradients
        normalized_losses = (losses - losses.mean()) / losses.std()
        g_hat = (self._epsilon.T @ normalized_losses).flatten()
        g_hat = g_hat / (self.popsize * self.sigma)
        self.params_vector -= self.lr * g_hat
        nn.utils.vector_to_parameters(self.params_vector, self.model.parameters())
