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
        use_rank_transform: bool,
    ):
        super().__init__(params, lr, popsize)
        self.popsize = popsize
        self.sigma = sigma
        self.use_antithetic_sampling = use_antithetic_sampling
        self.use_rank_transform = use_rank_transform
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
        if self.use_rank_transform:
            losses = losses.argsort().argsort() / (losses.shape[0] - 1) - 0.5
        normalized_losses = (losses - losses.mean()) / losses.std()
        g_hat = ((mutations.T / self.sigma) @ normalized_losses).flatten()
        g_hat = g_hat / (self.popsize * self.sigma)
        self.flat_params -= self.lr * g_hat
        nn.utils.vector_to_parameters(self.flat_params, self.params)


class SimpleEvolutionaryOptimizer(EvolutionaryOptimizer):
    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float,
        popsize: int,
        num_families: int,
        sigma: float,
    ):
        assert popsize % num_families == 0, "popsize must be a multiple of num_families"
        super().__init__(params, lr, popsize)
        self.num_families = num_families
        self.members_per_family = popsize // num_families
        self.sigma = sigma

        # every individual starts with the same parameters
        self.family_flat_params = nn.utils.parameters_to_vector(self.params).repeat(
            self.num_families, self.members_per_family, 1
        )

    def get_new_generation(self) -> tuple[torch.Tensor, torch.Tensor]:
        mutations = (
            torch.randn(
                self.num_families,
                self.members_per_family,
                self.family_flat_params.shape[2],
                device=self.family_flat_params.device,
                dtype=self.family_flat_params.dtype,
            )
            * self.sigma
        )
        # parents, ie the first member of each family should not be mutated
        mutations[:, 0, :] = 0

        mutated_batched_flat_params = (self.family_flat_params + mutations).flatten(0, 1)

        return mutated_batched_flat_params, mutations

    def step(self, losses: torch.Tensor, mutations: torch.Tensor):
        # get the flat indices of the lowest losses
        _, flat_indices = losses.flatten().topk(self.num_families, largest=False, sorted=True)

        # select the parents according to the flat_indices, new_parents of shape n_families x num_params
        new_parents = self.family_flat_params.flatten(0, 1)[flat_indices,]

        # unsqueezing turns into n_families x 1 x num_params. Then repeat along the 2nd dimension
        self.family_flat_params = new_parents.unsqueeze(1).repeat(1, self.members_per_family, 1)

        # load first parent (ie best one) into the params
        nn.utils.vector_to_parameters(new_parents[0], self.params)
