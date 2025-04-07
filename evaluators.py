from dataclasses import dataclass

import torch
import torch.nn as nn
import tqdm
from torch.utils.data import DataLoader

import loggers


@dataclass
class Evaluator1Config:
    device: torch.device
    do_log_models: bool


class Evaluator1:
    def __init__(self, model: nn.Module, dataloader: DataLoader, config: Evaluator1Config, logger: loggers.Logger):
        self.model = model
        self.dataloader = dataloader
        self.criterion = nn.CrossEntropyLoss()

        self.config = config
        self.logger = logger

    def evaluate_batch(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch
        x, y = x.to(self.config.device), y.to(self.config.device)

        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)

        predicted = y_hat.argmax(dim=-1)
        correct_preds = (predicted == y).sum().item()
        total_preds = y.shape[0]

        return loss, correct_preds, total_preds

    def evaluate_epoch(self, epoch: int):
        """Validate the current model"""
        self.model.eval()
        self.model.to(self.config.device)

        losses = torch.zeros(len(self.dataloader))
        correct_preds = torch.zeros(len(self.dataloader))
        total_preds = torch.zeros(len(self.dataloader))

        for batch_idx, batch in enumerate(tqdm.tqdm(self.dataloader, leave=False, desc=f"Epoch {epoch} - Validation")):
            batch_loss, batch_correct_preds, batch_total_preds = self.evaluate_batch(
                batch, batch_idx)
            losses[batch_idx] = batch_loss
            correct_preds[batch_idx] = batch_correct_preds
            total_preds[batch_idx] = batch_total_preds

        self.logger.log({"val/accuracy": correct_preds / total_preds}, epoch)
        self.logger.log({"val/loss": losses.mean().item()}, epoch)
        if self.config.do_log_models:
            self.logger.log_model(self.model)
