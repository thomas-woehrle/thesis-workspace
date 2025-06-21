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
        # find best params of this generation and load them into the model
        # we do this before any transformation of the losses
        best_mutation_idx = torch.argmin(losses)
        best_mutation = mutations[best_mutation_idx]
        best_params = self.mu + best_mutation
        nn.utils.vector_to_parameters(best_params, self.original_unflattened_params)

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
        popsize: int,
        sigma_init: float,
        use_antithetic_sampling: bool,
        use_rank_transform: bool,
        use_bdnes: bool = True,
    ):
        # turning into a list because params might be Iterator
        self.named_params = list(named_params)
        self.original_unflattened_params = [p for _, p in self.named_params]
        self.popsize = popsize
        self.use_antithetic_sampling = use_antithetic_sampling
        self.use_rank_transform = use_rank_transform
        self.use_bdnes = use_bdnes

        # this represents the distribution-based population (mu, sigma)
        self.mu = nn.utils.parameters_to_vector(self.original_unflattened_params).detach()
        # "trils" = triangular lower matrix
        self.cov_info = {}
        sigma_params = []

        for n, p in self.named_params:
            d = p.numel()
            if self.use_bdnes:
                if len(p.shape) == 4:  # Conv2d parameters
                    # Split Conv2d parameters into output channels
                    out_channels = p.shape[0]
                    for i in range(out_channels):
                        channel_params = p[i].flatten()
                        channel_d = channel_params.numel()
                        L = torch.full(
                            (channel_d, channel_d), 0.1 * sigma_init, device=self.mu.device
                        )
                        L = torch.tril(L)
                        L.fill_diagonal_(sigma_init)
                        tril_indices = torch.tril_indices(
                            row=channel_d, col=channel_d, offset=0, device=self.mu.device
                        )
                        flat_tril = L[tril_indices[0], tril_indices[1]]
                        self.cov_info[f"{n}_channel_{i}"] = {
                            "full_cov": True,
                            "flat_cov_tril": flat_tril,
                            "tril_indices": tril_indices,
                            "channel_index": i,
                            "param_name": n,
                        }
                        sigma_params.append(self.cov_info[f"{n}_channel_{i}"]["flat_cov_tril"])
                else:
                    # manage full cov variance for non-Conv2d parameters
                    L = torch.full((d, d), 0.1 * sigma_init, device=self.mu.device)
                    L = torch.tril(L)
                    L.fill_diagonal_(sigma_init)
                    tril_indices = torch.tril_indices(row=d, col=d, offset=0, device=self.mu.device)
                    flat_tril = L[tril_indices[0], tril_indices[1]]
                    self.cov_info[n] = {
                        "full_cov": True,
                        "flat_cov_tril": flat_tril,
                        "tril_indices": tril_indices,
                    }
                    sigma_params.append(self.cov_info[n]["flat_cov_tril"])
            else:
                # manage only sigma vector ie SNES
                sigma = torch.ones(d, device=self.mu.device) * sigma_init
                self.cov_info[n] = {
                    "full_cov": False,
                    "sigma": sigma,
                }
                sigma_params.append(self.cov_info[n]["sigma"])

        # create param groups to enable lr scheduling
        self.mu_param_group = {"params": self.mu, "lr": lr}
        self.sigma_param_group = {"params": sigma_params, "lr": sigma_lr}
        super().__init__([self.mu_param_group, self.sigma_param_group], {})

    def get_new_generation(self) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        mutations = []
        zs = []

        sample_size = self.popsize
        if self.use_antithetic_sampling:
            assert self.popsize % 2 == 0, "If using antithetic sampling, the popsize has to be even"
            sample_size = self.popsize // 2

        for n, p in self.named_params:
            d = p.numel()
            if self.use_bdnes:
                if len(p.shape) == 4:  # Conv2d parameters
                    out_channels = p.shape[0]
                    for i in range(out_channels):
                        channel_params = p[i].flatten()
                        channel_d = channel_params.numel()
                        z = torch.randn(
                            sample_size,
                            channel_d,
                            device=self.mu.device,
                        )
                        flat_tril = self.cov_info[f"{n}_channel_{i}"]["flat_cov_tril"]
                        tril_indices = self.cov_info[f"{n}_channel_{i}"]["tril_indices"]

                        L = torch.zeros(channel_d, channel_d, device=self.mu.device)
                        L[tril_indices[0], tril_indices[1]] = flat_tril

                        mutation = (L @ z.T).T
                        if self.use_antithetic_sampling:
                            mutation = torch.concatenate([mutation, -mutation], dim=0)
                            z = torch.concatenate([z, -z], dim=0)

                        mutations.append(mutation)
                        zs.append(z)
                else:
                    z = torch.randn(
                        sample_size,
                        d,
                        device=self.mu.device,
                    )
                    flat_tril = self.cov_info[n]["flat_cov_tril"]
                    tril_indices = self.cov_info[n]["tril_indices"]

                    L = torch.zeros(d, d, device=self.mu.device)
                    L[tril_indices[0], tril_indices[1]] = flat_tril

                    mutation = (L @ z.T).T
                    if self.use_antithetic_sampling:
                        mutation = torch.concatenate([mutation, -mutation], dim=0)
                        z = torch.concatenate([z, -z], dim=0)

                    mutations.append(mutation)
                    zs.append(z)
            else:
                z = torch.randn(
                    sample_size,
                    d,
                    device=self.mu.device,
                )
                mutation = self.cov_info[n]["sigma"] * z
                if self.use_antithetic_sampling:
                    mutation = torch.concatenate([mutation, -mutation], dim=0)
                    z = torch.concatenate([z, -z], dim=0)

                mutations.append(mutation)
                zs.append(z)

        mutations = torch.concatenate(mutations, dim=1)
        zs = torch.concatenate(zs, dim=1)

        # '+' operation broadcasts to (popsize, num_params)
        batched_mutated_flat_params = self.mu + mutations

        return batched_mutated_flat_params, mutations, zs

    def step(
        self, losses: torch.Tensor, mutations: torch.Tensor, zs: Optional[torch.Tensor] = None
    ):
        # losses of shape (popsize, )
        # estimate gradients
        if self.use_rank_transform:
            losses = losses.argsort().argsort() / (losses.shape[0] - 1) - 0.5

        # normalize losses
        normalized_losses = (losses - losses.mean()) / losses.std()

        # mu gradient is the same for all variants
        mu_grad = (mutations.T @ normalized_losses).flatten() / self.popsize
        self.mu -= self.mu_param_group["lr"] * mu_grad

        # update covariance parameters block by block
        current_idx = 0
        if zs is None:
            raise ValueError(
                "zs must not be None when using block-diagonal/full covariance updates."
            )
        for n, p in self.named_params:
            d = p.numel()
            if self.use_bdnes:
                if len(p.shape) == 4:  # Conv2d parameters
                    out_channels = p.shape[0]
                    for i in range(out_channels):
                        channel_params = p[i].flatten()
                        channel_d = channel_params.numel()
                        block_mutations = mutations[:, current_idx : current_idx + channel_d]
                        z = zs[:, current_idx : current_idx + channel_d]

                        g_L = torch.einsum("k,ki,kj->ij", normalized_losses, z, z)
                        g_L -= torch.sum(normalized_losses) * torch.eye(
                            channel_d, device=self.mu.device
                        )
                        g_L /= self.popsize
                        g_L_tril = torch.tril(g_L)

                        tril_indices = self.cov_info[f"{n}_channel_{i}"]["tril_indices"]
                        grad_flat_tril = g_L_tril[tril_indices[0], tril_indices[1]]

                        self.cov_info[f"{n}_channel_{i}"]["flat_cov_tril"] *= torch.exp(
                            -(self.sigma_param_group["lr"] / 2) * grad_flat_tril
                        )

                        current_idx += channel_d
                else:
                    block_mutations = mutations[:, current_idx : current_idx + d]
                    z = zs[:, current_idx : current_idx + d]

                    g_L = torch.einsum("k,ki,kj->ij", normalized_losses, z, z)
                    g_L -= torch.sum(normalized_losses) * torch.eye(d, device=self.mu.device)
                    g_L /= self.popsize
                    g_L_tril = torch.tril(g_L)

                    tril_indices = self.cov_info[n]["tril_indices"]
                    grad_flat_tril = g_L_tril[tril_indices[0], tril_indices[1]]

                    self.cov_info[n]["flat_cov_tril"] *= torch.exp(
                        -(self.sigma_param_group["lr"] / 2) * grad_flat_tril
                    )

                    current_idx += d
            else:
                block_mutations = mutations[:, current_idx : current_idx + d]
                sigma = self.cov_info[n]["sigma"]
                z = block_mutations / sigma
                sigma_grad = (((z**2) - 1).T @ normalized_losses).flatten() / self.popsize
                # update sigma using exponential step; '-' because we want to go in opposite direction of gradient
                self.cov_info[n]["sigma"] *= torch.exp(
                    -(self.sigma_param_group["lr"] / 2) * sigma_grad
                )

                current_idx += d

        # load mu into original params
        nn.utils.vector_to_parameters(self.mu, self.original_unflattened_params)
