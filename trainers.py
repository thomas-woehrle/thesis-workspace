from abc import ABC, abstractmethod
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader
from torch._functorch.apis import vmap

import loggers
import optimizers
import utils


class Trainer(ABC):
    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: optim.Optimizer,
        lr_scheduler: Optional[optim.lr_scheduler.LRScheduler],
        criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
        logger: loggers.Logger,
    ):
        self.model = model
        self.model.to(device=device, dtype=dtype)
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.criterion = criterion
        self.device = device
        self.dtype = dtype
        self.logger = logger

    @abstractmethod
    def train_step(self, x: torch.Tensor, y: torch.Tensor, batch_idx: int) -> torch.Tensor | float:
        """Can return float or tensor. Tensor should be preferred if train_epoch uses it as tensor anyway,
        which will often be the case. Reason is that the .item() conversion blocks the GPU for a very short piece of time."""
        raise NotImplementedError()

    def train_epoch(self, epoch: int):
        self.model.train()
        self.model.to(device=self.device, dtype=self.dtype)

        losses = torch.zeros(len(self.dataloader))
        for batch_idx, batch in enumerate(
            tqdm.tqdm(self.dataloader, leave=False, desc=f"Epoch {epoch} - Training")
        ):
            x, y = batch
            # x should be desired dtype, y should be long
            x, y = (
                x.to(device=self.device, dtype=self.dtype, non_blocking=True),
                y.to(device=self.device, dtype=torch.long, non_blocking=True),
            )
            losses[batch_idx] = self.train_step(x, y, batch_idx)

        self.logger.log({"train/loss": losses.mean().item()}, epoch)


class BackpropagationTrainer(Trainer):
    optimizer: optim.Optimizer

    def train_step(self, x: torch.Tensor, y: torch.Tensor, batch_idx: int) -> torch.Tensor | float:
        self.optimizer.zero_grad()

        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)

        loss.backward()
        self.optimizer.step()

        return loss

    def train_epoch(self, epoch: int):
        super().train_epoch(epoch)

        if self.lr_scheduler:
            self.logger.log({"debug/lr": self.lr_scheduler.get_last_lr()[0]}, epoch)
            self.lr_scheduler.step()


class EvolutionaryTrainer(Trainer):
    # TODO change to EvolutionaryOptimizer
    optimizer: optimizers.EvolutionaryOptimizer

    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: optimizers.EvolutionaryOptimizer,
        lr_scheduler: Optional[optim.lr_scheduler.LRScheduler],
        criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        use_parallel_forward_pass: bool,
        bn_track_running_stats: bool,
        use_instance_norm: bool,
        device: torch.device,
        dtype: torch.dtype,
        logger: loggers.Logger,
    ):
        super().__init__(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            criterion=criterion,
            device=device,
            dtype=dtype,
            logger=logger,
        )
        self.use_parallel_forward_pass = use_parallel_forward_pass
        self.bn_track_running_stats = bn_track_running_stats
        self.use_instance_norm = use_instance_norm
        self.batched_criterion = vmap(self.criterion, in_dims=(0, None))

    def train_step(self, x: torch.Tensor, y: torch.Tensor, batch_idx: int) -> torch.Tensor | float:
        new_generation_params, new_generation_buffers, mutations, mutated_flat_params = (
            self.optimizer.get_new_generation()
        )

        popsize = mutations.shape[0]

        if self.use_parallel_forward_pass:
            y_hat = utils.parallel_forward_pass(
                self.model, (new_generation_params, new_generation_buffers), x
            )
            losses = self.batched_criterion(y_hat, y)

        else:
            losses = torch.zeros(popsize, device=self.device)

            for i in range(popsize):
                nn.utils.vector_to_parameters(mutated_flat_params[i], self.model.parameters())
                y_hat = self.model(x)
                losses[i] = self.criterion(y_hat, y)

        self.optimizer.step(losses, mutations)
        return losses.mean()

    @torch.no_grad()
    def train_epoch(self, epoch: int):
        super().train_epoch(epoch)

        if (
            self.use_parallel_forward_pass
            and not self.use_instance_norm
            and self.bn_track_running_stats
        ):
            named_buffers = self.optimizer.get_current_buffers()
            self.model.load_state_dict(named_buffers, strict=False)


# @dataclass
# class SimpleEvolutionaryTrainerConfig(TrainerConfig):
#     n_families: int
#     members_per_family: int
#     sigma: float


# class SimpleEvolutionaryTrainer(Trainer[SimpleEvolutionaryTrainerConfig]):
#     # init as super()...

#     def __init__(
#         self,
#         model: nn.Module,
#         dataloader: DataLoader,
#         logger: loggers.Logger,
#         config: SimpleEvolutionaryTrainerConfig,
#     ):
#         super().__init__(model, dataloader, logger, config)
#         self.optimizer = optimizers.SimpleEvolutionaryOptimizer(
#             config.n_families,
#             config.members_per_family,
#             config.sigma,
#             self.model,
#             config.device,
#             config.dtype,
#         )
#         self.criterion = nn.CrossEntropyLoss()

#     def train_step(self, x: torch.Tensor, y: torch.Tensor, batch_idx: int) -> torch.Tensor | float:
#         self.optimizer.mutate()

#         losses = torch.zeros(self.optimizer.n_families, self.optimizer.members_per_family)
#         for family_idx in range(self.optimizer.n_families):
#             for member_idx in range(self.optimizer.members_per_family):
#                 self.optimizer.load_individual_into_model(family_idx, member_idx)
#                 y_hat = self.model(x)
#                 losses[family_idx][member_idx] = self.criterion(y_hat, y)

#         self.optimizer.step(losses)

#         return losses.mean()

#     @torch.no_grad()
#     def train_epoch(self, epoch: int):
#         return super().train_epoch(epoch)
