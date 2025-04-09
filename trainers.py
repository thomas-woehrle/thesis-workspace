from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler

import loggers


@dataclass
class TrainerConfig:
    device: torch.device


class Trainer[ConfigType: TrainerConfig](ABC):
    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        config: ConfigType,
        logger: loggers.Logger,
    ):
        self.model = model
        self.dataloader = dataloader
        self.config = config
        self.logger = logger

    @abstractmethod
    def train_batch(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        raise NotImplementedError()

    @abstractmethod
    def train_epoch(self, epoch: int):
        raise NotImplementedError


@dataclass
class OptimizerConfig:
    optimizer_name: str
    lr: float
    weight_decay: float = 0.0
    momentum: float = 0.0


@dataclass
class LRSchedulerConfig:
    use_cos_annealing_lr: bool
    T_max: Optional[int] = None
    eta_min: Optional[float] = None


@dataclass
class NormalTrainerConfig(TrainerConfig):
    device: torch.device
    optimizer_config: OptimizerConfig
    lr_scheduler_config: LRSchedulerConfig


def get_optimizer(model: nn.Module, optimizer_config: OptimizerConfig) -> Optimizer:
    if optimizer_config.optimizer_name == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=optimizer_config.lr,
            momentum=optimizer_config.momentum,
            weight_decay=optimizer_config.weight_decay,
        )
    elif optimizer_config.optimizer_name == "adam":
        return optim.Adam(
            model.parameters(),
            lr=optimizer_config.lr,
            weight_decay=optimizer_config.weight_decay,
        )
    elif optimizer_config.optimizer_name == "adamw":
        return optim.AdamW(
            model.parameters(),
            lr=optimizer_config.lr,
            weight_decay=optimizer_config.weight_decay,
        )
    else:
        raise ValueError(
            f"Optimizer {optimizer_config.optimizer_name} is not supported"
        )


def get_lr_scheduler(
    optimizer: Optimizer, scheduler_config: LRSchedulerConfig
) -> Optional[LRScheduler]:
    if scheduler_config.use_cos_annealing_lr:
        assert scheduler_config.eta_min is not None
        assert scheduler_config.T_max is not None
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, scheduler_config.T_max, scheduler_config.eta_min
        )
    else:
        return None


class NormalTrainer(Trainer[NormalTrainerConfig]):
    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        config: NormalTrainerConfig,
        logger: loggers.Logger,
    ):
        super().__init__(model, dataloader, config, logger)
        self.optimizer = get_optimizer(model, config.optimizer_config)
        self.lr_scheduler = get_lr_scheduler(self.optimizer, config.lr_scheduler_config)
        self.criterion = nn.CrossEntropyLoss()

    def train_batch(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        x, y = x.to(self.config.device), y.to(self.config.device)

        self.optimizer.zero_grad()

        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)

        loss.backward()
        self.optimizer.step()

        return loss

    def train_epoch(self, epoch: int):
        self.model.train()
        self.model.to(self.config.device)

        losses = torch.zeros(len(self.dataloader))
        for batch_idx, batch in enumerate(
            tqdm.tqdm(self.dataloader, leave=False, desc=f"Epoch {epoch} - Training")
        ):
            losses[batch_idx] = self.train_batch(batch, batch_idx)

        self.logger.log({"train/loss": losses.mean().item()}, epoch)

        if self.lr_scheduler:
            self.logger.log({"debug/lr": self.lr_scheduler.get_last_lr()[0]}, epoch)
            self.lr_scheduler.step()


class RandomEvolutionStrategy:
    def __init__(
        self, popsize: int, sigma: float, lr: float, initial_params_vector: torch.Tensor
    ):
        self.popsize = popsize
        self.sigma = sigma
        self.lr = lr
        self.params_vector = initial_params_vector
        self._last_epsilon = torch.zeros(popsize, len(initial_params_vector))

    def ask(self) -> torch.Tensor:
        epsilon = torch.randn(self.popsize, len(self.params_vector))
        self._last_epsilon = epsilon
        perturbations = epsilon * self.sigma
        solutions = perturbations + self.params_vector
        return solutions

    def tell(self, losses: torch.Tensor):
        # losses of shape popsize x 1
        # estimate gradients
        g_hat = (self._last_epsilon.T * losses).flatten()
        self.params_vector -= self.lr * g_hat


class EvolutionaryTrainerConfig(TrainerConfig):
    popsize: int
    sigma: float
    lr: float


class EvolutionaryTrainer(Trainer[EvolutionaryTrainerConfig]):
    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        config: EvolutionaryTrainerConfig,
        logger: loggers.Logger,
    ):
        super().__init__(model, dataloader, config, logger)
        self.es = RandomEvolutionStrategy(
            config.popsize,
            config.sigma,
            config.lr,
            nn.utils.parameters_to_vector(model.parameters()),
        )

    def train_batch(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        pass

    def train_epoch(self, epoch: int):
        pass
