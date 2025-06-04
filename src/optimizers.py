from abc import ABC, abstractmethod
from typing import Iterable
import torch
import torch.nn as nn
import torch.optim as optim

import utils


class EvolutionaryOptimizer(ABC, optim.Optimizer):
    popsize: int

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
        inner_optimizer_slug: str,
        momentum: float,
        weight_decay: float,
        use_antithetic_sampling: bool,
        use_rank_transform: bool,
    ):
        # turning into a list because params might be Iterator
        self.original_unflattened_params = list(params)
        self.sigma = sigma
        self.popsize = popsize
        self.use_antithetic_sampling = use_antithetic_sampling
        self.use_rank_transform = use_rank_transform

        # TODO create proper param_groups and pass them to the optimizer instead
        self.flat_params = nn.utils.parameters_to_vector(self.original_unflattened_params).detach()
        self.inner_optimizer = utils.get_non_evolutionary_optimizer(
            inner_optimizer_slug, [self.flat_params], lr, momentum, weight_decay
        )
        # expose the inner_optimizer param_groups to the outside, f.e. for lr scheduler
        self.param_groups = self.inner_optimizer.param_groups

        # nullify .grad, because they might start out as 'None'
        for p in self.original_unflattened_params:
            p.grad = torch.zeros_like(p)

    def get_new_generation(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.use_antithetic_sampling:
            assert self.popsize % 2 == 0, "If using antithetic sampling, the popsize has to be even"

            mutations = self.sigma * torch.randn(
                self.popsize // 2,
                len(self.flat_params),
                device=self.flat_params.device,
                dtype=self.flat_params.dtype,
            )
            mutations = torch.concatenate([mutations, -mutations], dim=0)
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
            losses = losses.to(self.flat_params.dtype)

        normalized_losses = (losses - losses.mean()) / losses.std()
        g_hat = ((mutations.T / self.sigma) @ normalized_losses).flatten()
        g_hat = g_hat / (self.popsize * self.sigma)

        # load gradients into .grad fields and step the inner optimizer
        self.inner_optimizer.zero_grad(set_to_none=False)
        self.flat_params.grad = g_hat
        self.inner_optimizer.step()

        # load updated flat params into original params
        nn.utils.vector_to_parameters(self.flat_params, self.original_unflattened_params)


class SNESOptimizer(EvolutionaryOptimizer):
    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float,
        popsize: int,
        sigma_init: float,
        inner_optimizer_slug: str,
        momentum: float,
        weight_decay: float,
        use_antithetic_sampling: bool,
        use_rank_transform: bool,
    ):
        # turning into a list because params might be Iterator
        self.original_unflattened_params = list(params)
        self.popsize = popsize
        self.use_antithetic_sampling = use_antithetic_sampling
        self.use_rank_transform = use_rank_transform

        # TODO create proper param_groups and pass them to the optimizer instead
        self.flat_params = nn.utils.parameters_to_vector(self.original_unflattened_params).detach()
        self.inner_optimizer = utils.get_non_evolutionary_optimizer(
            inner_optimizer_slug, [self.flat_params], lr, momentum, weight_decay
        )
        # expose the inner_optimizer param_groups to the outside, f.e. for lr scheduler
        self.param_groups = self.inner_optimizer.param_groups

        # afaict, the sigma can not be updated by standard optimizers, because of the local coordinate nature
        self.sigma = torch.ones_like(self.flat_params) * sigma_init
        # TODO remove this hardcoding
        self.sigma_lr = 1

        # nullify .grad, because they might start out as 'None'
        for p in self.original_unflattened_params:
            p.grad = torch.zeros_like(p)

    def get_new_generation(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.use_antithetic_sampling:
            assert self.popsize % 2 == 0, "If using antithetic sampling, the popsize has to be even"

            # self sigma of shape d, randn result of shape popsize // 2 x d; -> broadcast
            mutations = self.sigma * torch.randn(
                self.popsize // 2,
                len(self.flat_params),
                device=self.flat_params.device,
                dtype=self.flat_params.dtype,
            )
            mutations = torch.concatenate([mutations, -mutations], dim=0)
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
        # losses of shape (popsize, )
        # estimate gradients
        if self.use_rank_transform:
            losses = losses.argsort().argsort() / (losses.shape[0] - 1) - 0.5
            losses = losses.to(self.flat_params.dtype)

        # normalize losses
        normalized_losses = (losses - losses.mean()) / losses.std()

        # this is what is originally sampled from a gaussian; the s_k in the paper
        normalized_mutations = mutations / self.sigma

        flat_params_grad = (normalized_mutations.T @ normalized_losses).flatten() / self.popsize
        flat_params_grad *= self.sigma

        sigma_grad = (
            ((normalized_mutations**2) - 1).T @ normalized_losses
        ).flatten() / self.popsize

        # update flat_params using inner optimizer
        self.inner_optimizer.zero_grad(set_to_none=False)
        self.flat_params.grad = flat_params_grad
        self.inner_optimizer.step()

        # update sigma manually; '-' because we want to go in opposite direction of gradient
        self.sigma *= torch.exp(-(self.sigma_lr / 2) * sigma_grad)

        # load updated flat params into original params
        nn.utils.vector_to_parameters(self.flat_params, self.original_unflattened_params)
