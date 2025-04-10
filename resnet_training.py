import os
import random
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import wandb
import wandb.wandb_run

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

    dataloader = DataLoader(dataset, batch_size, num_workers=num_workers, shuffle=is_train)
    return dataloader


@dataclass
class TrainingConfig:
    trainer_config: trainers.TrainerConfig
    evaluator_config: evaluators.Evaluator1Config
    num_epochs: int
    batch_size: int
    use_img_transforms: bool
    use_evolution: bool
    model_slug: str
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


def run_training(training_config: TrainingConfig, wandb_run: Optional[wandb.wandb_run.Run]):
    if training_config.seed is not None:
        seed_everything(training_config.seed)

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

    if training_config.model_slug == "small_resnet20":
        model = resnet_cifar_small.resnet20(nb_cls=10 if training_config.is_cifar10 else 100)
    elif training_config.model_slug == "resnet18":
        model = resnet_cifar.ResNet18(nb_cls=10 if training_config.is_cifar10 else 100)
    else:
        raise ValueError(f"Model {training_config.model_slug} is not supported")
    logger = loggers.Logger(wandb_run)

    if training_config.use_evolution:
        assert isinstance(training_config.trainer_config, trainers.EvolutionaryTrainerConfig)
        trainer = trainers.EvolutionaryTrainer(
            model, train_dataloader, training_config.trainer_config, logger
        )
    else:
        assert isinstance(training_config.trainer_config, trainers.NormalTrainerConfig)
        trainer = trainers.NormalTrainer(
            model, train_dataloader, training_config.trainer_config, logger
        )
    evaluator = evaluators.Evaluator1(
        model, val_dataloader, training_config.evaluator_config, logger
    )

    for epoch in range(training_config.num_epochs):
        trainer.train_epoch(epoch)
        evaluator.evaluate_epoch(epoch)


if __name__ == "__main__":
    device = torch.device("mps")
    is_test_run = False
    is_cifar10 = False
    num_epochs = 10
    seed: Optional[int] = None
    use_evolution = False
    model_slug = "small_resnet20"

    if use_evolution:
        trainer_config = trainers.EvolutionaryTrainerConfig(
            device=device, popsize=100, sigma=0.1, lr=0.01, use_antithetic_sampling=True
        )
    else:
        trainer_config = trainers.NormalTrainerConfig(
            device=device,
            optimizer_config=trainers.OptimizerConfig(
                optimizer_name="sgd",
                lr=0.01 if is_test_run else 0.1,
                weight_decay=1e-4,
                momentum=0.9,
            ),
            lr_scheduler_config=trainers.LRSchedulerConfig(
                use_cos_annealing_lr=True, T_max=num_epochs, eta_min=0
            ),
        )

    evaluator_config = evaluators.Evaluator1Config(
        device=device, do_log_models=False if is_test_run else True
    )

    config = TrainingConfig(
        trainer_config=trainer_config,
        evaluator_config=evaluator_config,
        num_epochs=num_epochs,
        batch_size=16,
        use_evolution=use_evolution,
        use_img_transforms=True,
        use_data_subset=True if is_test_run else False,
        is_cifar10=is_cifar10,
        num_workers=0 if is_test_run else 4,
        seed=seed,
        model_slug=model_slug,
    )

    project = "thesis_baseline_testruns" if is_test_run else "thesis_baseline"

    with wandb.init(project=project, config=asdict(config)) as run:
        run_training(config, run)
