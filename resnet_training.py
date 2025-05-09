import os
import random
import sys
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import wandb
import wandb.wandb_run
import yaml

import evaluators
import loggers
import resnet_cifar
import resnet_cifar_small
import trainers

# Mean and standard deviation for CIFAR-100 (precomputed)
CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)


def get_cifar_dataloader(
    is_train: bool,
    batch_size: int,
    num_workers: int,
    use_every_nth: Optional[int],
    is_cifar10: bool,
    use_img_transforms: bool,
):
    train_transforms = (
        transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
            ]
        )
        if use_img_transforms
        else transforms.ToTensor()
    )

    val_transforms = (
        transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)]
        )
        if use_img_transforms
        else transforms.ToTensor()
    )

    if is_cifar10:
        dataset = torchvision.datasets.CIFAR10(
            "./artifacts",
            train=is_train,
            download=True,
            transform=train_transforms if is_train else val_transforms,
        )
    else:
        dataset = torchvision.datasets.CIFAR100(
            "./artifacts",
            train=is_train,
            download=True,
            transform=train_transforms if is_train else val_transforms,
        )

    if use_every_nth is not None:
        dataset = Subset(dataset, indices=range(0, len(dataset), use_every_nth))

    dataloader = DataLoader(
        dataset, batch_size, num_workers=num_workers, shuffle=is_train, pin_memory=True
    )
    return dataloader


@dataclass
class TrainingConfig:
    trainer_slug: str
    trainer_config: trainers.TrainerConfig
    evaluator_config: evaluators.Evaluator1Config
    num_epochs: int
    batch_size: int
    use_img_transforms: bool
    model_slug: str
    bn_track_running_stats: bool
    num_workers: int = 0
    use_data_subset: bool = False
    is_cifar10: bool = False
    seed: Optional[int] = None


def seed_everything(seed: int):
    """Seeds everything. Cuda seeding and benchmarking configuration excluded for now."""
    random.seed(seed)
    # Set PYTHONHASHSEED environment variable at a fixed value
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_trainer(
    trainer_slug: str,
    model: nn.Module,
    dataloader: DataLoader,
    logger: loggers.Logger,
    trainerConfig: trainers.TrainerConfig,
) -> trainers.Trainer:
    if trainer_slug == "backprop_trainer":
        assert isinstance(trainerConfig, trainers.NormalTrainerConfig)
        return trainers.NormalTrainer(model, dataloader, logger, trainerConfig)
    elif trainer_slug == "openai_evolutionary_trainer":
        assert isinstance(trainerConfig, trainers.OpenAIEvolutionaryTrainerConfig)
        return trainers.OpenAIEvolutionaryTrainer(model, dataloader, logger, trainerConfig)
    elif trainer_slug == "simple_evolutionary_trainer":
        assert isinstance(trainerConfig, trainers.SimpleEvolutionaryTrainerConfig)
        return trainers.SimpleEvolutionaryTrainer(model, dataloader, logger, trainerConfig)
    else:
        raise ValueError(f"Trainer {trainer_slug} is not supported")


def run_training(training_config: TrainingConfig, wandb_run: Optional[wandb.wandb_run.Run]):
    # Seed
    if training_config.seed is not None:
        seed_everything(training_config.seed)

    # Get dataloaders
    train_dataloader = get_cifar_dataloader(
        is_train=True,
        batch_size=training_config.batch_size,
        use_every_nth=1000 if training_config.use_data_subset else None,
        is_cifar10=training_config.is_cifar10,
        use_img_transforms=training_config.use_img_transforms,
        num_workers=training_config.num_workers,
    )
    val_dataloader = get_cifar_dataloader(
        is_train=False,
        batch_size=training_config.batch_size,
        use_every_nth=1000 if training_config.use_data_subset else None,
        is_cifar10=training_config.is_cifar10,
        use_img_transforms=training_config.use_img_transforms,
        num_workers=training_config.num_workers,
    )

    # Get model
    if training_config.model_slug == "small_resnet20":
        model = resnet_cifar_small.resnet20(
            bn_track_running_stats=training_config.bn_track_running_stats,
            nb_cls=10 if training_config.is_cifar10 else 100,
        )
    elif training_config.model_slug == "resnet18":
        model = resnet_cifar.ResNet18(nb_cls=10 if training_config.is_cifar10 else 100)
    else:
        raise ValueError(f"Model {training_config.model_slug} is not supported")

    model.to(
        device=training_config.trainer_config.device, dtype=training_config.trainer_config.dtype
    )
    if training_config.trainer_config.device == torch.device("mps"):
        model.compile(backend="aot_eager")
    else:
        model.compile()

    # Get logger
    logger = loggers.Logger(wandb_run)

    # Get trainer
    trainer = get_trainer(
        training_config.trainer_slug,
        model,
        train_dataloader,
        logger,
        training_config.trainer_config,
    )

    # Get evaluator
    evaluator = evaluators.Evaluator1(
        model, val_dataloader, training_config.evaluator_config, logger
    )

    # Run Training
    for epoch in range(training_config.num_epochs):
        trainer.train_epoch(epoch)
        evaluator.eval_epoch(epoch)


def load_config_from_yaml(config_path: str) -> tuple[TrainingConfig, str]:
    """Loads training configuration from a YAML file."""
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    # Handle device creation
    wandb_project = config_dict.pop("wandb_project")
    device = torch.device(config_dict.pop("device"))
    dtype = getattr(torch, config_dict.pop("dtype"))
    # .get and not .pop because used at multiple levels: not just trainer also model
    bn_track_running_stats = config_dict.get("bn_track_running_stats")

    # Handle TrainerConfig
    trainer_config_dict = config_dict.pop("trainer_config")

    if config_dict["trainer_slug"] == "openai_evolutionary_trainer":
        trainer_config = trainers.OpenAIEvolutionaryTrainerConfig(
            device=device,
            dtype=dtype,
            bn_track_running_stats=bn_track_running_stats,
            **trainer_config_dict,
        )
    elif config_dict["trainer_slug"] == "simple_evolutionary_trainer":
        trainer_config = trainers.SimpleEvolutionaryTrainerConfig(
            device=device,
            dtype=dtype,
            bn_track_running_stats=bn_track_running_stats,
            **trainer_config_dict,
        )
    elif config_dict["trainer_slug"] == "backprop_trainer":
        # Handle nested NormalTrainerConfig parts
        optimizer_config = trainers.OptimizerConfig(**trainer_config_dict["optimizer_config"])
        lr_scheduler_config = trainers.LRSchedulerConfig(
            **trainer_config_dict["lr_scheduler_config"]
        )

        trainer_config = trainers.NormalTrainerConfig(
            device=device,
            dtype=dtype,
            bn_track_running_stats=bn_track_running_stats,
            optimizer_config=optimizer_config,
            lr_scheduler_config=lr_scheduler_config,
        )

    # Handle EvaluatorConfig
    evaluator_config = evaluators.Evaluator1Config(
        device=device, dtype=dtype, **config_dict.pop("evaluator_config")
    )

    # Create main TrainingConfig
    training_config = TrainingConfig(
        trainer_config=trainer_config,
        evaluator_config=evaluator_config,
        **config_dict,  # Pass remaining top-level args
    )

    return training_config, wandb_project


if __name__ == "__main__":
    config_path = sys.argv[1]

    config, project = load_config_from_yaml(config_path)

    with wandb.init(project=project, config=asdict(config)) as run:
        run_training(config, run)
