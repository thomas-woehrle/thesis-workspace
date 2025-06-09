from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, RandomSampler, Subset


import evaluators
import loggers
import optimizers
import resnet_cifar_small
import resnet_cifar
import utils
import trainers


# --- General ---


@dataclass
class GeneralConfig:
    WANDB_PROJECT: str
    DEVICE: torch.device
    MP_DTYPE: torch.dtype
    SEED: Optional[int]
    NUM_TRAIN_STEPS: int
    TRAIN_INTERVAL_LENGTH: int
    USE_TORCH_COMPILE: bool
    CKPT_PATH: Optional[str] = None
    CKPT_EVERY_NTH_INTERVAL: Optional[int] = None
    RESUME_FROM_CKPT_PATH: Optional[str] = None


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


def get_cifar_dataloader(
    config: DataConfig,
    is_train: bool,
    num_train_steps: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
) -> DataLoader:
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

    if is_train:
        assert num_train_steps is not None, (
            "num_train_steps can not be None, when getting the train_dataloader"
        )
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=config.BATCH_SIZE,
            sampler=RandomSampler(
                dataset,
                replacement=False,
                num_samples=config.BATCH_SIZE * num_train_steps,
                generator=generator,
            ),
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
        )
    else:
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            shuffle=False,
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
        raise ValueError(utils.get_not_supported_message("Model", config.MODEL_SLUG))

    return model


# --- Trainer ---


@dataclass
class TrainerConfig:
    USE_PARALLEL_FORWARD_PASS: Optional[bool] = False


def get_trainer(
    config: TrainerConfig,
    model: nn.Module,
    dataloader: DataLoader,
    logger: loggers.Logger,
    optimizer: optim.Optimizer,
    lr_scheduler: Optional[optim.lr_scheduler.LRScheduler],
    use_instance_norm: bool,
    bn_track_running_stats: bool,
    use_torch_compile: Optional[bool],
    mp_dtype: torch.dtype,
) -> trainers.Trainer:
    criterion = nn.CrossEntropyLoss()

    if isinstance(optimizer, optimizers.EvolutionaryOptimizer):
        assert config.USE_PARALLEL_FORWARD_PASS is not None
        assert use_torch_compile is not None
        return trainers.EvolutionaryTrainer(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            criterion=criterion,
            use_parallel_forward_pass=config.USE_PARALLEL_FORWARD_PASS,
            bn_track_running_stats=bn_track_running_stats,
            use_instance_norm=use_instance_norm,
            mp_dtype=mp_dtype,
            logger=logger,
            use_torch_compile=use_torch_compile,
        )
    else:
        return trainers.BackpropagationTrainer(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            criterion=criterion,
            mp_dtype=mp_dtype,
            logger=logger,
        )
    # elif trainer_slug == "simple_evolutionary_trainer":
    #     assert isinstance(trainerConfig, trainers.SimpleEvolutionaryTrainerConfig)
    #     return trainers.SimpleEvolutionaryTrainer(model, dataloader, logger, trainerConfig)


# --- Optimizer ---


@dataclass
class OptimizerConfig:
    OPTIMIZER_SLUG: str
    IS_EVOLUTIONARY: bool
    LR: float
    SIGMA_LR: Optional[float] = None
    NES_INNER_OPTIMIZER_SLUG: Optional[str] = None
    WEIGHT_DECAY: float = 0.0
    MOMENTUM: float = 0.0
    POPSIZE: Optional[int] = None
    SIGMA_INIT: Optional[float] = None
    USE_ANTITHETIC_SAMPLING: Optional[bool] = False
    NUM_FAMILIES: Optional[int] = None
    USE_RANK_TRANSFORM: Optional[bool] = None


def get_optimizer(config: OptimizerConfig, model: nn.Module) -> optim.Optimizer:
    if config.IS_EVOLUTIONARY:
        if config.OPTIMIZER_SLUG == "openai_evolutionary_optimizer":
            assert config.POPSIZE is not None
            assert config.SIGMA_INIT is not None
            assert config.NES_INNER_OPTIMIZER_SLUG is not None
            assert config.USE_ANTITHETIC_SAMPLING is not None
            assert config.USE_RANK_TRANSFORM is not None

            return optimizers.OpenAIEvolutionaryOptimizer(
                model.parameters(),
                popsize=config.POPSIZE,
                sigma=config.SIGMA_INIT,
                lr=config.LR,
                inner_optimizer_slug=config.NES_INNER_OPTIMIZER_SLUG,
                momentum=config.MOMENTUM,
                weight_decay=config.WEIGHT_DECAY,
                use_antithetic_sampling=config.USE_ANTITHETIC_SAMPLING,
                use_rank_transform=config.USE_RANK_TRANSFORM,
            )
        elif config.OPTIMIZER_SLUG == "snes_optimizer":
            assert config.POPSIZE is not None
            assert config.SIGMA_INIT is not None
            assert config.NES_INNER_OPTIMIZER_SLUG is not None
            assert config.USE_ANTITHETIC_SAMPLING is not None
            assert config.USE_RANK_TRANSFORM is not None
            assert config.SIGMA_LR is not None

            return optimizers.SNESOptimizer(
                model.parameters(),
                popsize=config.POPSIZE,
                sigma_init=config.SIGMA_INIT,
                lr=config.LR,
                sigma_lr=config.SIGMA_LR,
                inner_optimizer_slug=config.NES_INNER_OPTIMIZER_SLUG,
                momentum=config.MOMENTUM,
                weight_decay=config.WEIGHT_DECAY,
                use_antithetic_sampling=config.USE_ANTITHETIC_SAMPLING,
                use_rank_transform=config.USE_RANK_TRANSFORM,
            )
        else:
            raise ValueError(utils.get_not_supported_message("Optimizer", config.OPTIMIZER_SLUG))
    else:
        return utils.get_non_evolutionary_optimizer(
            config.OPTIMIZER_SLUG,
            model.parameters(),
            config.LR,
            config.MOMENTUM,
            config.WEIGHT_DECAY,
        )


# --- LR Scheduler ---


@dataclass
class LRSchedulerConfig:
    LR_SCHEDULER_SLUG: Optional[str]
    T_MAX: Optional[int] = None
    ETA_MIN: Optional[float] = None
    MILESTONES: Optional[list[int]] = None
    GAMMA: Optional[float] = None


def get_lr_scheduler(
    config: LRSchedulerConfig, optimizer: optim.Optimizer
) -> Optional[optim.lr_scheduler.LRScheduler]:
    if config.LR_SCHEDULER_SLUG is None:
        return None
    elif config.LR_SCHEDULER_SLUG == "cosine_annealing":
        assert config.ETA_MIN is not None
        assert config.T_MAX is not None
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, config.T_MAX, config.ETA_MIN)
    elif config.LR_SCHEDULER_SLUG == "multi_step":
        assert config.MILESTONES is not None
        assert config.GAMMA is not None
        return optim.lr_scheduler.MultiStepLR(optimizer, config.MILESTONES, config.GAMMA)
    elif config.LR_SCHEDULER_SLUG == "multi_step":
        assert config.MILESTONES is not None
        assert config.GAMMA is not None
        return optim.lr_scheduler.MultiStepLR(optimizer, config.MILESTONES, config.GAMMA)
    else:
        raise ValueError(utils.get_not_supported_message("LR Scheduler", config.LR_SCHEDULER_SLUG))


# --- Evaluator ---


@dataclass
class EvaluatorConfig:
    DO_LOG_MODELS: bool


def get_evaluator(
    config: EvaluatorConfig,
    model: nn.Module,
    dataloader: DataLoader,
    logger: loggers.Logger,
) -> evaluators.Evaluator1:
    criterion = nn.CrossEntropyLoss()
    return evaluators.Evaluator1(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        do_log_models=config.DO_LOG_MODELS,
        logger=logger,
    )


# --- Complete ---


@dataclass
class RunConfig:
    general_config: GeneralConfig
    data_config: DataConfig
    model_config: ModelConfig
    trainer_config: TrainerConfig
    optimizer_config: OptimizerConfig
    lr_scheduler_config: LRSchedulerConfig
    evaluator_config: EvaluatorConfig
