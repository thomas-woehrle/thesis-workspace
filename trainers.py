from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from torch._functorch.apis import vmap
from torch.utils.data import DataLoader

import loggers
import optimizers


@dataclass
class TrainerConfig:
    device: torch.device
    dtype: torch.dtype
    track_bn_running_stats: bool


class Trainer[ConfigType: TrainerConfig](ABC):
    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        logger: loggers.Logger,
        config: ConfigType,
    ):
        self.model = model
        self.model.to(device=config.device, dtype=config.dtype)
        self.dataloader = dataloader
        self.logger = logger
        self.config = config

    @abstractmethod
    def train_step(self, x: torch.Tensor, y: torch.Tensor, batch_idx: int) -> torch.Tensor | float:
        """Can return float or tensor. Tensor should be preferred if train_epoch uses it as tensor anyway,
        which will often be the case. Reason is that the .item() conversion blocks the GPU for a very short piece of time."""
        raise NotImplementedError()

    def train_epoch(self, epoch: int):
        self.model.train()
        self.model.to(device=self.config.device, dtype=self.config.dtype)

        losses = torch.zeros(len(self.dataloader))
        for batch_idx, batch in enumerate(
            tqdm.tqdm(self.dataloader, leave=False, desc=f"Epoch {epoch} - Training")
        ):
            x, y = batch
            # x should be desired dtype, y should be long
            x, y = (
                x.to(device=self.config.device, dtype=self.config.dtype, non_blocking=True),
                y.to(device=self.config.device, dtype=torch.long, non_blocking=True),
            )
            losses[batch_idx] = self.train_step(x, y, batch_idx)

        self.logger.log({"train/loss": losses.mean().item()}, epoch)


@dataclass
class OptimizerConfig:
    optimizer_slug: str
    lr: float
    weight_decay: float = 0.0
    momentum: float = 0.0


@dataclass
class LRSchedulerConfig:
    lr_scheduler_slug: Optional[str]
    lr_scheduler_slug: Optional[str]
    T_max: Optional[int] = None
    eta_min: Optional[float] = None
    milestones: Optional[list[int]] = None
    gamma: Optional[float] = None
    milestones: Optional[list[int]] = None
    gamma: Optional[float] = None


@dataclass
class NormalTrainerConfig(TrainerConfig):
    optimizer_config: OptimizerConfig
    lr_scheduler_config: LRSchedulerConfig


def get_optimizer(model: nn.Module, optimizer_config: OptimizerConfig) -> optim.Optimizer:
    if optimizer_config.optimizer_slug == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=optimizer_config.lr,
            momentum=optimizer_config.momentum,
            weight_decay=optimizer_config.weight_decay,
        )
    elif optimizer_config.optimizer_slug == "adam":
        return optim.Adam(
            model.parameters(),
            lr=optimizer_config.lr,
            weight_decay=optimizer_config.weight_decay,
        )
    elif optimizer_config.optimizer_slug == "adamw":
        return optim.AdamW(
            model.parameters(),
            lr=optimizer_config.lr,
            weight_decay=optimizer_config.weight_decay,
        )
    else:
        raise ValueError(f"Optimizer {optimizer_config.optimizer_slug} is not supported")


def get_lr_scheduler(
    optimizer: optim.Optimizer, scheduler_config: LRSchedulerConfig
) -> Optional[optim.lr_scheduler.LRScheduler]:
    if scheduler_config.lr_scheduler_slug is None:
        return None
    elif scheduler_config.lr_scheduler_slug == "cosine_annealing":
        assert scheduler_config.eta_min is not None
        assert scheduler_config.T_max is not None
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, scheduler_config.T_max, scheduler_config.eta_min
        )
    elif scheduler_config.lr_scheduler_slug == "multi_step":
        assert scheduler_config.milestones is not None
        assert scheduler_config.gamma is not None
        return optim.lr_scheduler.MultiStepLR(
            optimizer, scheduler_config.milestones, scheduler_config.gamma
        )
    elif scheduler_config.lr_scheduler_slug == "multi_step":
        assert scheduler_config.milestones is not None
        assert scheduler_config.gamma is not None
        return optim.lr_scheduler.MultiStepLR(
            optimizer, scheduler_config.milestones, scheduler_config.gamma
        )
    else:
        raise ValueError(f"LR scheduler {scheduler_config.lr_scheduler_slug} is not supported")


class NormalTrainer(Trainer[NormalTrainerConfig]):
    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        logger: loggers.Logger,
        config: NormalTrainerConfig,
    ):
        super().__init__(model, dataloader, logger, config)
        self.optimizer = get_optimizer(model, config.optimizer_config)
        self.lr_scheduler = get_lr_scheduler(self.optimizer, config.lr_scheduler_config)
        self.criterion = nn.CrossEntropyLoss()

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


@dataclass
class OpenAIEvolutionaryTrainerConfig(TrainerConfig):
    popsize: int
    sigma: float
    lr: float
    use_antithetic_sampling: bool
    use_parallel_forward_pass: bool


class OpenAIEvolutionaryTrainer(Trainer[OpenAIEvolutionaryTrainerConfig]):
    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        logger: loggers.Logger,
        config: OpenAIEvolutionaryTrainerConfig,
    ):
        super().__init__(model, dataloader, logger, config)
        self.optimizer = optimizers.OpenAIEvolutionaryOptimizer(
            config.popsize,
            config.sigma,
            config.lr,
            self.model,
            config.use_antithetic_sampling,
            config.device,
            config.dtype,
        )
        self.criterion = nn.CrossEntropyLoss()
        self.multi_criterion = vmap(F.cross_entropy, in_dims=(0, None))

    def train_step(self, x: torch.Tensor, y: torch.Tensor, batch_idx: int) -> torch.Tensor | float:
        self.optimizer.prepare_mutations()

        if self.config.use_parallel_forward_pass:
            y_hat = self.optimizer.parallel_forward_pass(x)

            losses = self.multi_criterion(y_hat, y)
        else:
            losses = torch.zeros(self.config.popsize, device=self.config.device)

            for i in range(self.config.popsize):
                self.optimizer.load_mutation_into_model(i)
                y_hat = self.model(x)
                losses[i] = self.criterion(y_hat, y)

        self.optimizer.step(losses)
        return losses.mean()

    @torch.no_grad()
    def train_epoch(self, epoch: int):
        super().train_epoch(epoch)

        if self.config.use_parallel_forward_pass and self.config.track_bn_running_stats:
            # load buffers into model, only necessary to do this manually in the case of parallel pass
            named_buffers = self.optimizer.get_current_buffers()
            self.model.load_state_dict(named_buffers, strict=False)


@dataclass
class SimpleEvolutionaryTrainerConfig(TrainerConfig):
    n_families: int
    members_per_family: int
    sigma: float


class SimpleEvolutionaryTrainer(Trainer[SimpleEvolutionaryTrainerConfig]):
    # init as super()...

    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        logger: loggers.Logger,
        config: SimpleEvolutionaryTrainerConfig,
    ):
        super().__init__(model, dataloader, logger, config)
        self.optimizer = optimizers.SimpleEvolutionaryOptimizer(
            config.n_families,
            config.members_per_family,
            config.sigma,
            self.model,
            config.device,
            config.dtype,
        )
        self.criterion = nn.CrossEntropyLoss()

    def train_step(self, x: torch.Tensor, y: torch.Tensor, batch_idx: int) -> torch.Tensor | float:
        self.optimizer.mutate()

        losses = torch.zeros(self.optimizer.n_families, self.optimizer.members_per_family)
        for family_idx in range(self.optimizer.n_families):
            for member_idx in range(self.optimizer.members_per_family):
                self.optimizer.load_individual_into_model(family_idx, member_idx)
                y_hat = self.model(x)
                losses[family_idx][member_idx] = self.criterion(y_hat, y)

        self.optimizer.step(losses)

        return losses.mean()

    @torch.no_grad()
    def train_epoch(self, epoch: int):
        return super().train_epoch(epoch)
