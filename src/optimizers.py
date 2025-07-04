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


class BlockDiagonalNESOptimizer(EvolutionaryOptimizer):
    def __init__(
        self,
        named_params: Iterable[tuple[str, torch.Tensor]],
        lr: float,
        sigma_lr: float,
        B_lr: float,
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
        # B matrices
        self.cov_shape_info: dict[str, torch.Tensor] = {}
        # sigma scalars
        self.cov_scale_info: dict[str, torch.Tensor] = {}

        # create param groups to enable lr scheduling
        self.mu_param_group = {"params": self.mu, "lr": lr}
        self.sigma_param_group = {"params": [], "lr": sigma_lr}
        self.B_param_group = {"params": [], "lr": B_lr}
        super().__init__([self.mu_param_group, self.sigma_param_group, self.B_param_group], {})

        self._initialize_cov_info(sigma_init)

    def _initialize_cov_info(self, sigma_init: float):
        for n, p in self.named_params:
            if len(p.shape) == 4:  # Conv2d parameters
                out_channels, in_channels, kh, kw = p.shape
                channel_d = in_channels * kh * kw

                # Directly initialize sigma and B for stability
                sigmas = torch.ones(out_channels, device=self.mu.device) * sigma_init
                Bs = (
                    torch.eye(channel_d, device=self.mu.device)
                    .expand(out_channels, channel_d, channel_d)
                    .clone()
                )

                self.cov_scale_info[n] = sigmas
                self.cov_shape_info[n] = Bs
                self.sigma_param_group["params"].append(sigmas)
                self.B_param_group["params"].append(Bs)
            else:
                # manage full cov variance for non-Conv2d parameters
                d = p.numel()

                # Directly initialize sigma and B for stability
                sigma = torch.tensor(sigma_init, device=self.mu.device)
                B = torch.eye(d, device=self.mu.device)

                self.cov_scale_info[n] = sigma
                self.cov_shape_info[n] = B
                self.sigma_param_group["params"].append(sigma)
                self.B_param_group["params"].append(B)

    def get_new_generation(self) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        mutations = []
        normal_samples = []

        sample_size = self.popsize
        if self.use_antithetic_sampling:
            assert self.popsize % 2 == 0, "If using antithetic sampling, the popsize has to be even"
            sample_size = self.popsize // 2

        for n, p in self.named_params:
            if len(p.shape) == 4:  # Batched Conv2d parameters
                Bs = self.cov_shape_info[n]  # shape (out_channels, channel_d, channel_d)
                sigmas = self.cov_scale_info[n]  # shape (out_channels,)
                num_blocks, d, _ = Bs.shape

                # s_b shape: (sample_size, num_blocks, d)
                s_b = torch.randn(sample_size, num_blocks, d, device=self.mu.device)
                # bmm requires (b, n, m) @ (b, m, p), so we permute
                # s_bmm shape: (num_blocks, sample_size, d)
                s_bmm = s_b.permute(1, 0, 2)
                mutations_bmm = torch.bmm(s_bmm, Bs)
                # mutations_b back to shape (sample_size, num_blocks, d)
                mutations_b = mutations_bmm.permute(1, 0, 2)
                # apply sigma scale, sigmas broadcasts to match shape
                mutations_b *= sigmas.view(1, -1, 1)

                # flatten the block dimension for concatenation later
                # shape: (sample_size, num_blocks * d)
                mutation = mutations_b.reshape(sample_size, -1)
                s = s_b.reshape(sample_size, -1)

                if self.use_antithetic_sampling:
                    mutation = torch.cat([mutation, -mutation], dim=0)
                    s = torch.cat([s, -s], dim=0)

                mutations.append(mutation)
                normal_samples.append(s)
            else:  # Un-batched other parameters
                B = self.cov_shape_info[n]
                sigma = self.cov_scale_info[n]
                s = torch.randn(sample_size, B.shape[0], device=self.mu.device)
                mutation = sigma * (s @ B)

                if self.use_antithetic_sampling:
                    mutation = torch.cat([mutation, -mutation], dim=0)
                    s = torch.cat([s, -s], dim=0)

                mutations.append(mutation)
                normal_samples.append(s)

        mutations = torch.cat(mutations, dim=1)
        normal_samples = torch.cat(normal_samples, dim=1)

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
            if len(p.shape) == 4:  # Batched Conv2d parameters
                out_channels, in_channels, kh, kw = p.shape
                channel_d = in_channels * kh * kw
                num_params_in_block = out_channels * channel_d

                Bs = self.cov_shape_info[n]  # shape (out_channels, channel_d, channel_d)
                sigmas = self.cov_scale_info[n]  # shape (out_channels,)

                # --- mu update ---
                mu_slice = self.mu[current_idx : current_idx + num_params_in_block]
                g_delta_slice = g_delta[current_idx : current_idx + num_params_in_block]
                # g_delta_reshaped shape: (out_channels, 1, channel_d)
                g_delta_reshaped = g_delta_slice.view(out_channels, 1, channel_d)

                update_b = torch.bmm(g_delta_reshaped, Bs)
                # apply sigma scale, sigmas broadcasts to match shape
                update_b *= sigmas.view(-1, 1, 1)
                update = self.mu_param_group["lr"] * update_b.flatten()
                mu_slice.sub_(update)

                # --- sigma and B update ---
                s_slice = normal_samples[:, current_idx : current_idx + num_params_in_block]
                # s_reshaped shape: (popsize, out_channels, channel_d)
                s_reshaped = s_slice.view(self.popsize, out_channels, channel_d)
                # s_for_einsum shape: (out_channels, popsize, channel_d) for einsum
                s_for_einsum = s_reshaped.permute(1, 0, 2)

                # g_M_b shape: (out_channels, channel_d, channel_d)
                g_M_b = torch.einsum(
                    "k,bki,bkj->bij", normalized_losses, s_for_einsum, s_for_einsum
                )
                g_M_b -= torch.sum(normalized_losses) * torch.eye(
                    channel_d, device=self.mu.device
                ).expand_as(Bs)
                g_M_b /= self.popsize

                # Decompose g_M into g_sigma and g_B
                # g_sigma_b shape: (out_channels,)
                g_sigma_b = torch.diagonal(g_M_b, dim1=-2, dim2=-1).sum(dim=-1) / channel_d
                # g_B_b shape: (out_channels, channel_d, channel_d)
                g_B_b = g_M_b - g_sigma_b.view(-1, 1, 1) * torch.eye(
                    channel_d, device=self.mu.device
                ).expand_as(Bs)

                # Update sigma
                sigmas *= torch.exp(-(self.sigma_param_group["lr"] / 2) * g_sigma_b)

                # Update B
                matrix_exp_b = torch.matrix_exp(-(self.B_param_group["lr"] / 2) * g_B_b)
                Bs.copy_(torch.bmm(Bs, matrix_exp_b))

                current_idx += num_params_in_block
            else:  # Un-batched other parameters
                d = p.numel()
                B = self.cov_shape_info[n]
                sigma = self.cov_scale_info[n]

                # --- mu update ---
                mu_slice = self.mu[current_idx : current_idx + d]
                g_delta_slice = g_delta[current_idx : current_idx + d]
                update = self.mu_param_group["lr"] * sigma * (g_delta_slice @ B)
                mu_slice.sub_(update)

                # --- sigma and B update ---
                s_slice = normal_samples[:, current_idx : current_idx + d]
                g_M = torch.einsum("k,ki,kj->ij", normalized_losses, s_slice, s_slice)
                g_M -= torch.sum(normalized_losses) * torch.eye(d, device=self.mu.device)
                g_M /= self.popsize

                # Decompose g_M into g_sigma and g_B
                g_sigma = torch.trace(g_M) / d
                g_B = g_M - g_sigma * torch.eye(d, device=self.mu.device)

                # Update sigma
                sigma *= torch.exp(-(self.sigma_param_group["lr"] / 2) * g_sigma)

                # Update B
                matrix_exp = torch.matrix_exp(-(self.B_param_group["lr"] / 2) * g_B)
                B.copy_(B @ matrix_exp)

                current_idx += d

        # load mu into original params
        nn.utils.vector_to_parameters(self.mu, self.original_unflattened_params)
