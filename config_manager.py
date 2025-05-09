from dataclasses import dataclass
from typing import Optional

import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

import resnet_cifar_small
import resnet_cifar

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


# ------


# --- Model ---


@dataclass
class ModelConfig:
    MODEL_SLUG: bool
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

    return model


# ------
