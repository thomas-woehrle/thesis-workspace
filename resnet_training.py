from dataclasses import asdict, dataclass
import os
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, Subset
import wandb
import wandb.wandb_run


@dataclass
class Config:
    is_dry_run: bool = False
    num_epochs: int = 10
    do_log_models: bool = True


def get_cifar100_dataloader(is_train: bool, batch_size: int, use_every_nth: int | None = None):
    dataset = torchvision.datasets.CIFAR100(
        "./artifacts", train=is_train, download=True, transform=torchvision.transforms.ToTensor())
    if use_every_nth is not None:
        dataset = Subset(dataset, indices=range(
            0, len(dataset), use_every_nth))

    dataloader = DataLoader(dataset, batch_size, shuffle=is_train)
    return dataloader


class ResNetTraining:
    def __init__(self, config: Config, wandb_run: (wandb.wandb_run.Run | None) = None):
        torch.manual_seed(42)
        self.run = wandb_run
        self.model = torchvision.models.resnet18()
        self.config = config
        self.curr_ep = 0
        self.train_dataloader = get_cifar100_dataloader(
            is_train=True, batch_size=16, use_every_nth=1000 if config.is_dry_run else None)
        self.val_dataloader = get_cifar100_dataloader(
            is_train=False, batch_size=16, use_every_nth=1000 if config.is_dry_run else None)
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.CrossEntropyLoss()

    def train(self):
        """Trains for one epoch"""
        self.model.train()

        losses = torch.zeros(len(self.train_dataloader))
        for idx, (x, y) in enumerate(self.train_dataloader):
            self.optimizer.zero_grad()

            y_hat = self.model(x)
            loss = self.criterion(y_hat, y)
            losses[idx] = loss

            loss.backward()
            self.optimizer.step()

        self.log({"train/loss": losses.mean().item()}, self.curr_ep)

    @torch.no_grad()
    def validate(self):
        """Validate the current model"""
        self.model.eval()

        losses = torch.zeros(len(self.val_dataloader))
        for idx, (x, y) in enumerate(self.val_dataloader):
            y_hat = self.model(x)
            loss = self.criterion(y_hat, y)
            losses[idx] = loss

        self.log({"val/loss": losses.mean().item()}, self.curr_ep)
        if self.config.do_log_models:
            self.log_model()

    def run_training(self):
        for ep in range(self.config.num_epochs):
            self.curr_ep = ep
            self.train()
            self.validate()

    def log(self, data: dict[str, Any], step: (int | None) = None, commit: (bool | None) = None):
        """Effectively mirrors wandb run.log API, see https://docs.wandb.ai/ref/python/run/#log"""
        if self.run:
            self.run.log(data, step, commit)
        else:
            raise NotImplementedError()

    def log_model(self, name: (str | None) = None, aliases: (list[str] | None) = None):
        """Effectively mirrors wandb run.log_model API"""
        path = "./temp_model"
        torch.save(self.model.state_dict(), path)
        if self.run:
            self.run.log_model(path, name, aliases)
            os.remove(path)
        else:
            raise NotImplementedError()


if __name__ == "__main__":
    config = Config(
        is_dry_run=True
    )

    with wandb.init(project="thesis_baseline", group="dry_runs", config=asdict(config)) as run:
        training = ResNetTraining(config, run)
        training.run_training()
