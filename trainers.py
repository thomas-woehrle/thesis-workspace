from abc import ABC, abstractmethod
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader

import optimizers
import loggers
from torch._functorch.apis import vmap


class Trainer(ABC):
    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: optim.Optimizer | optimizers.EvolutionaryOptimizer,
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


class NormalTrainer(Trainer):
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


# @dataclass
# class OpenAIEvolutionaryTrainerConfig(TrainerConfig):
#     popsize: int
#     sigma: float
#     lr: float
#     use_antithetic_sampling: bool
#     use_parallel_forward_pass: bool


class EvolutionaryTrainer(Trainer):
    # TODO change to EvolutionaryOptimizer
    optimizer: optimizers.OpenAIEvolutionaryOptimizer

    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: optimizers.EvolutionaryOptimizer,
        lr_scheduler: Optional[optim.lr_scheduler.LRScheduler],
        criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
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
        self.batched_criterion = vmap(self.criterion, in_dims=(0, None))

    def train_step(self, x: torch.Tensor, y: torch.Tensor, batch_idx: int) -> torch.Tensor | float:
        self.optimizer.prepare_mutations()

        # if self.config.use_parallel_forward_pass:
        y_hat = self.optimizer.parallel_forward_pass(x)

        losses = self.batched_criterion(y_hat, y)
        # else:
        #     losses = torch.zeros(self.config.popsize, device=self.config.device)

        #     for i in range(self.config.popsize):
        #         self.optimizer.load_mutation_into_model(i)
        #         y_hat = self.model(x)
        #         losses[i] = self.criterion(y_hat, y)

        self.optimizer.step(losses)
        return losses.mean()

    @torch.no_grad()
    def train_epoch(self, epoch: int):
        super().train_epoch(epoch)

        # if self.config.use_parallel_forward_pass and self.config.bn_track_running_stats:
        # load buffers into model, only necessary to do this manually in the case of parallel pass
        named_buffers = self.optimizer.get_current_buffers()
        self.model.load_state_dict(named_buffers, strict=False)


class OpenAIEvolutionaryTrainer(EvolutionaryTrainer):
    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: optimizers.EvolutionaryOptimizer,
        lr_scheduler: Optional[optim.lr_scheduler.LRScheduler],
        criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
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
        self.multi_criterion = torch.vmap(self.criterion, in_dims=(0, None))

    # def train_step(self, x: torch.Tensor, y: torch.Tensor, batch_idx: int) -> torch.Tensor | float:
    #     self.optimizer.prepare_mutations()

    #     if self.config.use_parallel_forward_pass:
    #         y_hat = self.optimizer.parallel_forward_pass(x)

    #         losses = self.multi_criterion(y_hat, y)
    #     else:
    #         losses = torch.zeros(self.config.popsize, device=self.config.device)

    #         for i in range(self.config.popsize):
    #             self.optimizer.load_mutation_into_model(i)
    #             y_hat = self.model(x)
    #             losses[i] = self.criterion(y_hat, y)

    #     self.optimizer.step(losses)
    #     return losses.mean()

    # @torch.no_grad()
    # def train_epoch(self, epoch: int):
    #     super().train_epoch(epoch)

    #     if self.config.use_parallel_forward_pass and self.config.bn_track_running_stats:
    #         # load buffers into model, only necessary to do this manually in the case of parallel pass
    #         named_buffers = self.optimizer.get_current_buffers()
    #         self.model.load_state_dict(named_buffers, strict=False)


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
