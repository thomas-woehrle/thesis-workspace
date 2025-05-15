from abc import ABC, abstractmethod
from typing import Iterable
import torch
import torch.nn as nn
import torch.optim as optim


class EvolutionaryOptimizer(ABC, optim.Optimizer):
    def __init__(self, params: Iterable[torch.Tensor], lr: float, popsize: int, **kwargs):
        defaults = dict(lr=lr, popsize=popsize, **kwargs)
        super().__init__(params, defaults)

    @abstractmethod
    def get_new_generation(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def step(self, losses: torch.Tensor, mutations: torch.Tensor):
        pass


class OpenAIEvolutionaryOptimizer(EvolutionaryOptimizer):
    """Can only handle one param group currently"""

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float,
        popsize: int,
        sigma: float,
        use_antithetic_sampling: bool,
        use_rank_transform: bool,
        weight_decay: float,
    ):
        assert weight_decay >= 0
        super().__init__(
            params,
            lr,
            popsize,
            sigma=sigma,
            use_antithetic_sampling=use_antithetic_sampling,
            use_rank_transform=use_rank_transform,
            weight_decay=weight_decay,
        )
        self.flat_params = nn.utils.parameters_to_vector(self.param_groups[0]["params"])
        self.params_dtype: torch.dtype = self.param_groups[0]["params"][0].dtype

    def get_new_generation(self) -> tuple[torch.Tensor, torch.Tensor]:
        param_group = self.param_groups[0]

        if param_group["use_antithetic_sampling"]:
            assert param_group["popsize"] % 2 == 0, (
                "If using antithetic sampling, the popsize has to be even"
            )

            mutations = param_group["sigma"] * torch.randn(
                param_group["popsize"] // 2,
                len(self.flat_params),
                device=self.flat_params.device,
                dtype=self.flat_params.dtype,
            )
            mutations = torch.concatenate([mutations, mutations], dim=0)
        else:
            mutations = param_group["sigma"] * torch.randn(
                param_group["popsize"],
                len(self.flat_params),
                device=self.flat_params.device,
                dtype=self.flat_params.dtype,
            )

        # '+' operation broadcasts to (popsize, num_params)
        batched_mutated_flat_params = self.flat_params + mutations

        return batched_mutated_flat_params, mutations

    def step(self, losses: torch.Tensor, mutations: torch.Tensor):
        # losses of shape popsize x 1
        param_group = self.param_groups[0]

        if param_group["use_rank_transform"]:
            losses = losses.argsort().argsort() / (losses.shape[0] - 1) - 0.5
            losses = losses.to(self.params_dtype)

        # Weight decay
        if param_group["weight_decay"] != 0.0:
            # switch to fp32 because of precision
            flat_params_fp32 = self.flat_params.float()
            flat_params_fp32.mul_(1 - param_group["lr"] * param_group["weight_decay"])
            self.flat_params = flat_params_fp32.to(self.params_dtype)

        # Gradient estimation
        normalized_losses = (losses - losses.mean()) / losses.std()
        g_hat = ((mutations.T / param_group["sigma"]) @ normalized_losses).flatten()
        g_hat = g_hat / (param_group["popsize"] * param_group["sigma"])

        self.flat_params -= param_group["lr"] * g_hat
        nn.utils.vector_to_parameters(self.flat_params, param_group["params"])


class SimpleEvolutionaryOptimizer(EvolutionaryOptimizer):
    """Can only handle one param group currently"""

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float,
        popsize: int,
        sigma: float,
        num_families: int,
    ):
        assert popsize % num_families == 0, "popsize must be a multiple of num_families"
        super().__init__(
            params,
            lr,
            popsize,
            sigma=sigma,
            num_families=num_families,
            members_per_family=popsize // num_families,
        )

        self.param_groups[0]["params"]
        # every individual starts with the same parameters
        self.family_flat_params = nn.utils.parameters_to_vector(
            self.param_groups[0]["params"]
        ).repeat(
            self.param_groups[0]["num_families"], self.param_groups[0]["members_per_family"], 1
        )

    def get_new_generation(self) -> tuple[torch.Tensor, torch.Tensor]:
        param_group = self.param_groups[0]
        mutations = (
            torch.randn(
                param_group["num_families"],
                param_group["members_per_family"],
                self.family_flat_params.shape[2],
                device=self.family_flat_params.device,
                dtype=self.family_flat_params.dtype,
            )
            * param_group["sigma"]
        )
        # parents, ie the first member of each family should not be mutated
        mutations[:, 0, :] = 0

        mutated_batched_flat_params = (self.family_flat_params + mutations).flatten(0, 1)

        return mutated_batched_flat_params, mutations

    def step(self, losses: torch.Tensor, mutations: torch.Tensor):
        # get the flat indices of the lowest losses
        param_group = self.param_groups[0]
        _, flat_indices = losses.flatten().topk(
            param_group["num_families"], largest=False, sorted=True
        )

        # select the parents according to the flat_indices, new_parents of shape n_families x num_params
        new_parents = self.family_flat_params.flatten(0, 1)[flat_indices,]

        # unsqueezing turns into n_families x 1 x num_params. Then repeat along the 2nd dimension
        self.family_flat_params = new_parents.unsqueeze(1).repeat(
            1, param_group["members_per_family"], 1
        )

        # load first parent (ie best one) into the params
        nn.utils.vector_to_parameters(new_parents[0], param_group["params"])
