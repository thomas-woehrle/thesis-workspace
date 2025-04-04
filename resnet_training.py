from dataclasses import asdict, dataclass
import os
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, Subset
import tqdm
import wandb
import wandb.wandb_run

import es


@dataclass
class Config:
    is_dry_run: bool = False
    num_epochs: int = 10
    do_log_models: bool = True
    device: torch.device = torch.device("cpu")
    use_evolution: bool = False


def get_cifar100_dataloader(is_train: bool, batch_size: int, use_every_nth: int | None = None):
    dataset = torchvision.datasets.CIFAR100(
        "./artifacts", train=is_train, download=True, transform=torchvision.transforms.ToTensor())
    if use_every_nth is not None:
        dataset = Subset(dataset, indices=range(
            0, len(dataset), use_every_nth))

    dataloader = DataLoader(dataset, batch_size, shuffle=is_train)
    return dataloader


def get_resnet_for_cifar100() -> nn.Module:
    # make other versions than resnet18 available in the future
    model = torchvision.models.resnet18()
    model.fc = nn.Linear(512, 100)
    return model


class ResNetTraining:
    def __init__(self, config: Config, wandb_run: (wandb.wandb_run.Run | None) = None):
        torch.manual_seed(42)
        self.run = wandb_run
        self.model = get_resnet_for_cifar100().to(config.device)
        self.config = config
        self.curr_ep = 0
        self.train_dataloader = get_cifar100_dataloader(
            is_train=True, batch_size=16, use_every_nth=1000 if config.is_dry_run else None)
        self.val_dataloader = get_cifar100_dataloader(
            is_train=False, batch_size=16, use_every_nth=1000 if config.is_dry_run else None)
        self.es_solver = es.RandomEvolutionSolver(
            dict(self.model.named_parameters()), device=config.device, popsize=10, sigma=0.001, alpha=0.00001) if config.use_evolution else None
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.CrossEntropyLoss()

    def _load_named_params_into_model(self, named_params: dict[str, torch.Tensor]):
        """Loads the given parameters, but keeps the current buffers"""
        self.model.load_state_dict(
            {**named_params, **dict(self.model.named_buffers())})

    def _evaluate_mutation(self, mutated_weights: dict[str, torch.Tensor], x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self._load_named_params_into_model(mutated_weights)

        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        return loss

    def train(self):
        """Trains for one epoch"""
        self.model.train()

        losses = torch.zeros(len(self.train_dataloader))
        for idx, (x, y) in enumerate(tqdm.tqdm(self.train_dataloader, leave=False, desc=f"Epoch {self.curr_ep} - Training")):
            x, y = x.to(self.config.device), y.to(self.config.device)

            if self.config.use_evolution:
                assert self.es_solver is not None

                solutions = self.es_solver.ask()

                solution_losses = torch.zeros(self.es_solver.popsize)
                for mutation_idx in range(self.es_solver.popsize):
                    solution_losses[mutation_idx] = self._evaluate_mutation(
                        solutions[mutation_idx], x, y)

                losses[idx] = solution_losses.mean()

                self.es_solver.tell(solution_losses)
                # after the telling, the parameters inside the solver are updated
                self._load_named_params_into_model(
                    self.es_solver.named_parameters)
            else:
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
        correct_preds = 0
        total_preds = 0

        for idx, (x, y) in enumerate(tqdm.tqdm(self.val_dataloader, leave=False, desc=f"Epoch {self.curr_ep} - Validation")):
            x, y = x.to(self.config.device), y.to(self.config.device)

            y_hat = self.model(x)
            loss = self.criterion(y_hat, y)
            losses[idx] = loss

            predicted = y_hat.argmax(dim=-1)
            correct_preds += (predicted == y).sum().item()
            total_preds += y.shape[0]

        self.log({"val/accuracy": correct_preds / total_preds}, self.curr_ep)
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
        if self.run is not None:
            self.run.log(data, step, commit)
        else:
            print(f"Step {step}: {data}")

    def log_model(self, name: (str | None) = None, aliases: (list[str] | None) = None):
        """Effectively mirrors wandb run.log_model API"""
        path = "./temp_model"
        torch.save(self.model.state_dict(), path)
        if self.run is not None:
            self.run.log_model(path, name, aliases)
            os.remove(path)
        else:
            raise NotImplementedError()


if __name__ == "__main__":
    config = Config(
        num_epochs=2,
        # device=torch.device(
        #     "mps") if torch.mps.is_available() else torch.device("cpu"),
        is_dry_run=True,
        do_log_models=False,
        use_evolution=True
    )

    with wandb.init(project="thesis_baseline_dryruns", config=asdict(config)) as run:
        training = ResNetTraining(config, run)
        training.run_training()
