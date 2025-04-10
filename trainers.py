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
    def train_batch(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> float:
        raise NotImplementedError()

    def train_epoch(self, epoch: int):
        self.model.train()
        self.model.to(self.config.device)

        losses = torch.zeros(len(self.dataloader))
        for batch_idx, batch in enumerate(
            tqdm.tqdm(self.dataloader, leave=False, desc=f"Epoch {epoch} - Training")
        ):
            losses[batch_idx] = self.train_batch(batch, batch_idx)

        self.logger.log({"train/loss": losses.mean().item()}, epoch)


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
        raise ValueError(f"Optimizer {optimizer_config.optimizer_name} is not supported")


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

    def train_batch(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        x, y = x.to(self.config.device), y.to(self.config.device)

        self.optimizer.zero_grad()

        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train_epoch(self, epoch: int):
        super().train_epoch(epoch)

        if self.lr_scheduler:
            self.logger.log({"debug/lr": self.lr_scheduler.get_last_lr()[0]}, epoch)
            self.lr_scheduler.step()


class RandomEvolutionStrategy:
    def __init__(
        self,
        popsize: int,
        sigma: float,
        lr: float,
        initial_params_vector: torch.Tensor,
        use_antithetic_sampling: bool,
        device: torch.device,
    ):
        self.popsize = popsize
        self.sigma = sigma
        self.lr = lr
        self.use_antithetic_sampling = use_antithetic_sampling
        self.params_vector = initial_params_vector
        self._epsilon = torch.zeros(popsize, len(initial_params_vector))
        self.device = device

    def sample_new_epsilon(self):
        """Creates new epsilon. Epsilon is of shape popsize * num_params"""
        if self.use_antithetic_sampling:
            assert self.popsize % 2 == 0, "If using antithetic sampling, the popsize has to be even"

            # This seemingly weird direct assignment to self._epsilon is done to not waste RAM
            self._epsilon = torch.randn(
                self.popsize // 2, len(self.params_vector), device=self.device
            )
            self._epsilon = torch.concatenate([self._epsilon, -self._epsilon], dim=0)
        else:
            self._epsilon = torch.randn(self.popsize, len(self.params_vector), device=self.device)

    def ask(self, individual_idx: int):
        return self.params_vector + self._epsilon[individual_idx] * self.sigma

    def tell(self, losses: torch.Tensor):
        # losses of shape popsize x 1
        # estimate gradients
        g_hat = (self._epsilon.T @ (losses - losses.mean())).flatten()
        g_hat = g_hat / (self.popsize * self.sigma)
        self.params_vector -= self.lr * g_hat


@dataclass
class EvolutionaryTrainerConfig(TrainerConfig):
    popsize: int
    sigma: float
    lr: float
    use_antithetic_sampling: bool


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
            nn.utils.parameters_to_vector(model.parameters()).to(config.device),
            config.use_antithetic_sampling,
            device=config.device,
        )
        self.criterion = nn.CrossEntropyLoss()

    def train_batch(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        # Adapted from https://github.com/hardmaru/estool/tree/master
        x, y = batch
        x, y = x.to(self.config.device), y.to(self.config.device)

        self.es.sample_new_epsilon()
        losses = torch.zeros(self.es.popsize, device=self.config.device)

        with torch.no_grad():
            for i in range(self.es.popsize):
                solution_i = self.es.ask(i)
                nn.utils.vector_to_parameters(solution_i, self.model.parameters())
                y_hat = self.model(x)
                losses[i] = self.criterion(y_hat, y)

        self.es.tell(losses)
        return losses.mean().item()

    @torch.no_grad()
    def train_epoch(self, epoch: int):
        return super().train_epoch(epoch)
