from dataclasses import dataclass
import os
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import tqdm
import wandb.wandb_run


@dataclass
class NormalTrainerConfig:
    device: torch.device
    num_epochs: int


class Logger:
    def __init__(self, wandb_run: Optional[wandb.wandb_run.Run]):
        self.wandb_run = wandb_run

    def log(self, data: dict[str, Any], step: Optional[int] = None, commit: Optional[bool] = None):
        """Effectively mirrors wandb run.log API, see https://docs.wandb.ai/ref/python/run/#log"""
        if self.wandb_run is not None:
            self.wandb_run.log(data, step, commit)
        else:
            print(f"Step {step}: {data}")

    def log_model(self, model: nn.Module, name: (str | None) = None, aliases: (list[str] | None) = None):
        """Effectively mirrors wandb run.log_model API"""
        path = "./temp_model"
        torch.save(model.state_dict(), path)
        if self.wandb_run is not None:
            self.wandb_run.log_model(path, name, aliases)
            os.remove(path)
        else:
            raise NotImplementedError()


class NormalTrainer:
    def __init__(self, model: nn.Module, dataloader: DataLoader, config: NormalTrainerConfig, logger: Logger):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optim.Adam(model.parameters())
        self.criterion = nn.CrossEntropyLoss()

        self.config = config
        self.logger = logger

    def train_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
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

        losses = torch.zeros(len(self.dataloader))
        for batch_idx, batch in enumerate(tqdm.tqdm(self.dataloader, leave=False, desc=f"Epoch {epoch} - Training")):
            losses[batch_idx] = self.train_step(batch, batch_idx)

        self.logger.log({"train/loss": losses.mean().item()}, epoch)
