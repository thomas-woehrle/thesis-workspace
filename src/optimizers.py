from abc import ABC, abstractmethod
from typing import Any, Iterable, Optional
import torch
import torch.nn as nn
import torch.optim as optim

import utils


class EvolutionaryOptimizer(ABC, optim.Optimizer):
    popsize: int
    sigma: float | torch.Tensor

    @abstractmethod
    def get_new_generation(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        pass

    @abstractmethod
    def step(
        self, losses: torch.Tensor, mutations: torch.Tensor, zs: Optional[torch.Tensor] = None
    ):
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
        defaults = {}
        super().__init__(params, defaults)
        # turning into a list because params might be Iterator
        self.original_unflattened_params = list(self.param_groups[0]["params"])
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

    def get_new_generation(self) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if self.use_antithetic_sampling:
            assert self.popsize % 2 == 0, "If using antithetic sampling, the popsize has to be even"

            mutations = self.sigma * torch.randn(
                self.popsize // 2,
                len(self.flat_params),
                device=self.flat_params.device,
            )
            mutations = torch.concatenate([mutations, -mutations], dim=0)
        else:
            mutations = self.sigma * torch.randn(
                self.popsize,
                len(self.flat_params),
                device=self.flat_params.device,
            )

        # '+' operation broadcasts to (popsize, num_params)
        batched_mutated_flat_params = self.flat_params + mutations

        return batched_mutated_flat_params, mutations, None

    def step(
        self, losses: torch.Tensor, mutations: torch.Tensor, zs: Optional[torch.Tensor] = None
    ):
        # losses of shape popsize x 1
        # estimate gradients
        if self.use_rank_transform:
            losses = losses.argsort().argsort() / (losses.shape[0] - 1) - 0.5

        # normalize losses
        normalized_losses = (losses - losses.mean()) / losses.std()

        # this is what is originally sampled from a gaussian; the s_k in the paper
        normalized_mutations = mutations / self.sigma

        flat_params_grad = (normalized_mutations.T @ normalized_losses).flatten() / self.popsize
        flat_params_grad *= self.sigma

        # load gradients into .grad fields and step the inner optimizer
        self.inner_optimizer.zero_grad(set_to_none=False)
        self.flat_params.grad = flat_params_grad
        self.inner_optimizer.step()

        # load updated flat params into original params
        nn.utils.vector_to_parameters(self.flat_params, self.original_unflattened_params)


class SNESOptimizer(EvolutionaryOptimizer):
    sigma: torch.Tensor

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float,
        sigma_lr: float,
        popsize: int,
        sigma_init: float,
        use_antithetic_sampling: bool,
        use_rank_transform: bool,
        adaptation_sampling_factor: Optional[float],
        adaptation_sampling_c_prime: Optional[float],
    ):
        # turning into a list because params might be Iterator
        self.original_unflattened_params = list(params)
        self.popsize = popsize
        self.use_antithetic_sampling = use_antithetic_sampling
        self.use_rank_transform = use_rank_transform
        self.adaptation_sampling_factor = adaptation_sampling_factor
        self.adaptation_sampling_c_prime = adaptation_sampling_c_prime

        # this represents the distribution-based population (mu, sigma)
        self.mu = nn.utils.parameters_to_vector(self.original_unflattened_params).detach()
        self.sigma = torch.ones_like(self.mu) * sigma_init
        self.sigma_adaptation_sampled = self.sigma.clone().detach()

        # create param groups to enable lr scheduling
        self.mu_param_group = {"params": self.mu, "lr": lr}
        self.sigma_param_group = {"params": self.sigma, "lr": sigma_lr}
        super().__init__([self.mu_param_group, self.sigma_param_group], {})

    def get_new_generation(
        self, from_adaptation_sampled_sigma: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        sigma = self.sigma_adaptation_sampled if from_adaptation_sampled_sigma else self.sigma

        if self.use_antithetic_sampling:
            assert self.popsize % 2 == 0, "If using antithetic sampling, the popsize has to be even"

            # self sigma of shape d, randn result of shape popsize // 2 x d; -> broadcast
            mutations = sigma * torch.randn(
                self.popsize // 2,
                len(self.mu),
                device=self.mu.device,
            )
            mutations = torch.concatenate([mutations, -mutations], dim=0)
        else:
            mutations = sigma * torch.randn(
                self.popsize,
                len(self.mu),
                device=self.mu.device,
            )

        # '+' operation broadcasts to (popsize, num_params)
        batched_mutated_flat_params = self.mu + mutations

        return batched_mutated_flat_params, mutations, None

    def step(
        self, losses: torch.Tensor, mutations: torch.Tensor, zs: Optional[torch.Tensor] = None
    ):
        # losses of shape (popsize, )
        # estimate gradients
        if self.use_rank_transform:
            losses = losses.argsort().argsort() / (losses.shape[0] - 1) - 0.5

        # normalize losses
        normalized_losses = (losses - losses.mean()) / losses.std()

        # this is what is originally sampled from a gaussian; the s_k in the paper
        normalized_mutations = mutations / self.sigma

        mu_grad = (normalized_mutations.T @ normalized_losses).flatten() / self.popsize
        mu_grad *= self.sigma

        sigma_grad = (
            ((normalized_mutations**2) - 1).T @ normalized_losses
        ).flatten() / self.popsize

        # update mu using gradient descent step
        self.mu -= self.mu_param_group["lr"] * mu_grad

        # create adaptation sampled sigma
        if self.adaptation_sampling_factor is not None:
            self.sigma_adaptation_sampled = self.sigma * torch.exp(
                -(self.sigma_param_group["lr"] * self.adaptation_sampling_factor / 2) * sigma_grad
            )
        # update sigma using exponential step; '-' because we want to go in opposite direction of gradient
        self.sigma *= torch.exp(-(self.sigma_param_group["lr"] / 2) * sigma_grad)

        # load mu into original params
        nn.utils.vector_to_parameters(self.mu, self.original_unflattened_params)

    def state_dict(self) -> dict[str, Any]:
        return {
            "mu": self.mu,
            "mu_lr": self.mu_param_group["lr"],
            "sigma": self.sigma,
            "sigma_lr": self.sigma_param_group["lr"],
        }

    def load_state_dict(self, state_dict: dict):
        self.mu.copy_(state_dict["mu"])
        self.mu_param_group["lr"] = state_dict["mu_lr"]
        self.sigma.copy_(state_dict["sigma"])
        self.sigma_param_group["lr"] = state_dict["sigma_lr"]


def get_initial_A(d: int, diag_init: float, device: torch.device):
    A = torch.ones(d, d, device=device) * 1e-4
    A.fill_diagonal_(diag_init)
    return A


def sample_from_A(sample_size: int, A: torch.Tensor, device: torch.device):
    s = torch.randn(sample_size, A.shape[0], device=device)
    mutation = s @ A
    return mutation, s


def step_mu(mu: torch.Tensor, g_delta: torch.Tensor, A: torch.Tensor, lr: float):
    update = lr * g_delta @ A
    mu.sub_(update)


def step_A(A: torch.Tensor, losses: torch.Tensor, s: torch.Tensor, lr: float):
    popsize = losses.shape[0]
    # g_M is the gradient for the covariance matrix
    g_M = torch.einsum("k,ki,kj->ij", losses, s, s)
    g_M -= torch.sum(losses) * torch.eye(g_M.shape[0], device=g_M.device)
    g_M /= popsize

    matrix_exp = torch.matrix_exp(-0.5 * lr * g_M)
    A.data.copy_(A @ matrix_exp)


class BlockDiagonalNESOptimizer(EvolutionaryOptimizer):
    def __init__(
        self,
        named_params: Iterable[tuple[str, torch.Tensor]],
        lr: float,
        sigma_lr: float,
        popsize: int,
        sigma_init: float,
        use_antithetic_sampling: bool,
        use_rank_transform: bool,
    ):
        # turning into a list because params might be Iterator
        self.named_params = list(named_params)
        self.original_unflattened_params = [p for _, p in self.named_params]
        self.popsize = popsize
        self.use_antithetic_sampling = use_antithetic_sampling
        self.use_rank_transform = use_rank_transform

        # this represents the distribution-based population
        self.mu = nn.utils.parameters_to_vector(self.original_unflattened_params).detach()
        self.cov_info: dict[str, torch.Tensor] = {}

        # create param groups to enable lr scheduling
        self.mu_param_group = {"params": self.mu, "lr": lr}
        self.sigma_param_group = {"params": [], "lr": sigma_lr}
        super().__init__([self.mu_param_group, self.sigma_param_group], {})

        self._initialize_cov_info(sigma_init)

    def _initialize_cov_info(self, sigma_init: float):
        for n, p in self.named_params:
            d = p.numel()
            if len(p.shape) == 4:  # Conv2d parameters
                # Split Conv2d parameters into output channels
                out_channels = p.shape[0]
                for i in range(out_channels):
                    channel_params = p[i].flatten()
                    channel_d = channel_params.numel()
                    A = get_initial_A(channel_d, sigma_init, self.mu.device)
                    self.cov_info[f"{n}_channel_{i}"] = A
                    self.sigma_param_group["params"].append(A)
            else:
                # manage full cov variance for non-Conv2d parameters
                A = get_initial_A(d, sigma_init, self.mu.device)
                self.cov_info[n] = A
                self.sigma_param_group["params"].append(A)

    def get_new_generation(self) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        mutations = []
        normal_samples = []

        sample_size = self.popsize
        if self.use_antithetic_sampling:
            assert self.popsize % 2 == 0, "If using antithetic sampling, the popsize has to be even"
            sample_size = self.popsize // 2

        for n, p in self.named_params:
            if len(p.shape) == 4:  # Conv2d parameters
                out_channels = p.shape[0]
                for i in range(out_channels):
                    A = self.cov_info[f"{n}_channel_{i}"]
                    mutation, s = sample_from_A(sample_size, A, self.mu.device)

                    if self.use_antithetic_sampling:
                        mutation = torch.concatenate([mutation, -mutation], dim=0)
                        s = torch.concatenate([s, -s], dim=0)

                    mutations.append(mutation)
                    normal_samples.append(s)
            else:
                A = self.cov_info[n]
                mutation, s = sample_from_A(sample_size, A, self.mu.device)

                if self.use_antithetic_sampling:
                    mutation = torch.concatenate([mutation, -mutation], dim=0)
                    s = torch.concatenate([s, -s], dim=0)

                mutations.append(mutation)
                normal_samples.append(s)

        mutations = torch.concatenate(mutations, dim=1)
        normal_samples = torch.concatenate(normal_samples, dim=1)

        # '+' operation broadcasts to (popsize, num_params)
        batched_mutated_flat_params = self.mu + mutations

        return batched_mutated_flat_params, mutations, normal_samples

    def step(self, losses: torch.Tensor, mutations: torch.Tensor, normal_samples: torch.Tensor):
        # losses are of shape (popsize, )
        if self.use_rank_transform:
            losses = losses.argsort().argsort() / (losses.shape[0] - 1) - 0.5

        # normalize losses
        normalized_losses = (losses - losses.mean()) / losses.std()

        # delta gradient. Used to update mu later on
        g_delta = (normalized_losses @ normal_samples).flatten() / self.popsize

        # update covariance parameters block by block
        current_idx = 0
        for n, p in self.named_params:
            d = p.numel()
            if len(p.shape) == 4:  # Conv2d parameters
                out_channels = p.shape[0]
                for i in range(out_channels):
                    channel_params = p[i].flatten()
                    channel_d = channel_params.numel()
                    A = self.cov_info[f"{n}_channel_{i}"]

                    # update mu
                    step_mu(
                        self.mu[current_idx : current_idx + channel_d],
                        g_delta[current_idx : current_idx + channel_d],
                        A,
                        self.mu_param_group["lr"],
                    )

                    # update A
                    step_A(
                        A,
                        normalized_losses,
                        normal_samples[:, current_idx : current_idx + channel_d],
                        self.sigma_param_group["lr"],
                    )

                    current_idx += channel_d
            else:
                A = self.cov_info[n]

                # update mu
                step_mu(
                    self.mu[current_idx : current_idx + d],
                    g_delta[current_idx : current_idx + d],
                    A,
                    self.mu_param_group["lr"],
                )

                # update A
                step_A(
                    A,
                    normalized_losses,
                    normal_samples[:, current_idx : current_idx + d],
                    self.sigma_param_group["lr"],
                )
                current_idx += d

        # load mu into original params
        nn.utils.vector_to_parameters(self.mu, self.original_unflattened_params)
