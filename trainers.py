from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import tqdm

import loggers


@dataclass
class NormalTrainerConfig:
    device: torch.device
    num_epochs: int


class NormalTrainer:
    def __init__(self, model: nn.Module, dataloader: DataLoader, config: NormalTrainerConfig, logger: loggers.Logger):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optim.Adam(model.parameters())
        self.criterion = nn.CrossEntropyLoss()

        self.config = config
        self.logger = logger

    def train_batch(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
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
            losses[batch_idx] = self.train_batch(batch, batch_idx)

        self.logger.log({"train/loss": losses.mean().item()}, epoch)
