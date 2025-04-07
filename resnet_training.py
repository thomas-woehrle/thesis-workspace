from dataclasses import asdict, dataclass
from typing import Optional

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Subset
import wandb
import wandb.wandb_run

import evaluators
import loggers
import resnet
import trainers


def get_cifar100_dataloader(is_train: bool, batch_size: int, use_every_nth: int | None = None):
    dataset = torchvision.datasets.CIFAR100(
        "./artifacts", train=is_train, download=True, transform=torchvision.transforms.ToTensor())
    if use_every_nth is not None:
        dataset = Subset(dataset, indices=range(
            0, len(dataset), use_every_nth))

    dataloader = DataLoader(dataset, batch_size, shuffle=is_train)
    return dataloader


@dataclass
class TrainingConfig:
    trainer_config: trainers.NormalTrainerConfig
    evaluator_config: evaluators.Evaluator1Config
    num_epochs: int
    batch_size: int
    do_use_data_subset: bool = False


def run_training(training_config: TrainingConfig, wandb_run: Optional[wandb.wandb_run.Run]):
    train_dataloader = get_cifar100_dataloader(
        is_train=True,
        batch_size=training_config.batch_size,
        use_every_nth=1000 if training_config.do_use_data_subset else None)
    val_dataloader = get_cifar100_dataloader(
        is_train=False,
        batch_size=training_config.batch_size,
        use_every_nth=1000 if training_config.do_use_data_subset else None)

    model = resnet.resnet20()
    logger = loggers.Logger(wandb_run)

    trainer = trainers.NormalTrainer(
        model, train_dataloader, training_config.trainer_config, logger)
    evaluator = evaluators.Evaluator1(
        model, val_dataloader, training_config.evaluator_config, logger)

    for epoch in range(training_config.num_epochs):
        trainer.train_epoch(epoch)
        evaluator.evaluate_epoch(epoch)


if __name__ == "__main__":
    device = torch.device("mps")
    is_test_run = True

    trainer_config = trainers.NormalTrainerConfig(
        device=device
    )
    evaluator_config = evaluators.Evaluator1Config(
        device=device,
        do_log_models=False if is_test_run else True
    )

    config = TrainingConfig(
        trainer_config=trainer_config,
        evaluator_config=evaluator_config,
        num_epochs=20,
        batch_size=32,
        do_use_data_subset=True if is_test_run else False
    )

    project = "thesis_baseline_testruns" if is_test_run else "thesis_baseline"

    with wandb.init(project=project, config=asdict(config)) as run:
        run_training(config, run)
