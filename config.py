from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

import evaluators
import loggers
import resnet_cifar_small
import resnet_cifar
import trainers


# --- Data ---


@dataclass
class DataConfig:
    BATCH_SIZE: int
    NUM_WORKERS: int
    USE_EVERY_NTH: Optional[int]
    IS_CIFAR10: bool
    USE_IMG_TRANSFORMS: bool


# Mean and standard deviation for CIFAR-100 (precomputed)
CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)


def get_cifar_dataloader(config: DataConfig, is_train: bool) -> DataLoader:
    train_transforms = (
        transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
            ]
        )
        if config.USE_IMG_TRANSFORMS
        else transforms.ToTensor()
    )

    val_transforms = (
        transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)]
        )
        if config.USE_IMG_TRANSFORMS
        else transforms.ToTensor()
    )

    if config.IS_CIFAR10:
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

    if config.USE_EVERY_NTH is not None:
        dataset = Subset(dataset, indices=range(0, len(dataset), config.USE_EVERY_NTH))

    dataloader = DataLoader(
        dataset,
        config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=is_train,
        pin_memory=True,
    )

    return dataloader


# --- Model ---


@dataclass
class ModelConfig:
    MODEL_SLUG: str
    USE_INSTANCE_NORM: bool
    BN_TRACK_RUNNING_STATS: bool


def get_model(config: ModelConfig, is_cifar10: bool) -> nn.Module:
    if config.MODEL_SLUG == "small_resnet20":
        model = resnet_cifar_small.resnet20(
            nb_cls=10 if is_cifar10 else 100,
            use_instance_norm=config.USE_INSTANCE_NORM,
            bn_track_running_stats=config.BN_TRACK_RUNNING_STATS,
        )
    elif config.MODEL_SLUG == "resnet18":
        model = resnet_cifar.ResNet18(nb_cls=10 if is_cifar10 else 100)
    else:
        raise ValueError(f"Model {config.MODEL_SLUG} is not supported")

    return model


# --- Trainer ---


@dataclass
class TrainerConfig:
    TRAINER_SLUG: str


def get_trainer(
    config: TrainerConfig,
    model: nn.Module,
    dataloader: DataLoader,
    logger: loggers.Logger,
    optimizer: optim.Optimizer,
    lr_scheduler: Optional[optim.lr_scheduler.LRScheduler],
    device: torch.device,
    dtype: torch.dtype,
) -> trainers.Trainer:
    criterion = nn.CrossEntropyLoss()

    if config.TRAINER_SLUG == "backprop_trainer":
        return trainers.NormalTrainer(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            criterion=criterion,
            device=device,
            dtype=dtype,
            logger=logger,
        )
    # elif trainer_slug == "openai_evolutionary_trainer":
    #     assert isinstance(trainerConfig, trainers.OpenAIEvolutionaryTrainerConfig)
    #     return trainers.OpenAIEvolutionaryTrainer(model, dataloader, logger, trainerConfig)
    # elif trainer_slug == "simple_evolutionary_trainer":
    #     assert isinstance(trainerConfig, trainers.SimpleEvolutionaryTrainerConfig)
    #     return trainers.SimpleEvolutionaryTrainer(model, dataloader, logger, trainerConfig)
    else:
        raise ValueError(f"Trainer {config.TRAINER_SLUG} is not supported")


# --- Optimizer ---


@dataclass
class OptimizerConfig:
    OPTIMIZER_SLUG: str
    LR: float
    WEIGHT_DECAY: float = 0.0
    MOMENTUM: float = 0.0


def get_optimizer(config: OptimizerConfig, model: nn.Module) -> optim.Optimizer:
    if config.OPTIMIZER_SLUG == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=config.LR,
            momentum=config.MOMENTUM,
            weight_decay=config.WEIGHT_DECAY,
        )
    elif config.OPTIMIZER_SLUG == "adam":
        return optim.Adam(
            model.parameters(),
            lr=config.LR,
            weight_decay=config.WEIGHT_DECAY,
        )
    elif config.OPTIMIZER_SLUG == "adamw":
        return optim.AdamW(
            model.parameters(),
            lr=config.LR,
            weight_decay=config.WEIGHT_DECAY,
        )
    else:
        raise ValueError(f"Optimizer {config.OPTIMIZER_SLUG} is not supported")


# --- LR Scheduler ---


@dataclass
class LRSchedulerConfig:
    LR_SCHEDULER_SLUG: Optional[str]
    T_MAX: Optional[int] = None
    ETA_MIN: Optional[float] = None
    MILESTONES: Optional[list[int]] = None
    GAMMA: Optional[float] = None


def get_lr_scheduler(
    optimizer: optim.Optimizer, scheduler_config: LRSchedulerConfig
) -> Optional[optim.lr_scheduler.LRScheduler]:
    if scheduler_config.LR_SCHEDULER_SLUG is None:
        return None
    elif scheduler_config.LR_SCHEDULER_SLUG == "cosine_annealing":
        assert scheduler_config.ETA_MIN is not None
        assert scheduler_config.T_MAX is not None
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, scheduler_config.T_MAX, scheduler_config.ETA_MIN
        )
    elif scheduler_config.LR_SCHEDULER_SLUG == "multi_step":
        assert scheduler_config.MILESTONES is not None
        assert scheduler_config.GAMMA is not None
        return optim.lr_scheduler.MultiStepLR(
            optimizer, scheduler_config.MILESTONES, scheduler_config.GAMMA
        )
    elif scheduler_config.LR_SCHEDULER_SLUG == "multi_step":
        assert scheduler_config.MILESTONES is not None
        assert scheduler_config.GAMMA is not None
        return optim.lr_scheduler.MultiStepLR(
            optimizer, scheduler_config.MILESTONES, scheduler_config.GAMMA
        )
    else:
        raise ValueError(f"LR scheduler {scheduler_config.LR_SCHEDULER_SLUG} is not supported")


# --- Evaluator ---


@dataclass
class EvaluatorConfig:
    DO_LOG_MODELS: bool


def get_evaluator(
    config: EvaluatorConfig,
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    dtype: torch.dtype,
    logger: loggers.Logger,
) -> evaluators.Evaluator1:
    criterion = nn.CrossEntropyLoss()
    return evaluators.Evaluator1(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        device=device,
        dtype=dtype,
        do_log_models=config.DO_LOG_MODELS,
        logger=logger,
    )


# --- General ---


@dataclass
class GeneralConfig:
    WANDB_PROJECT: str
    DEVICE: torch.device
    DTYPE: torch.dtype
    SEED: Optional[int]
    NUM_EPOCHS: int


@dataclass
class RunConfig:
    general_config: GeneralConfig
    data_config: DataConfig
    model_config: ModelConfig
    optimizer_config: OptimizerConfig
    lr_scheduler_config: LRSchedulerConfig
    trainer_config: TrainerConfig
    evaluator_config: EvaluatorConfig
