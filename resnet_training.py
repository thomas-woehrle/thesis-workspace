import os
import random
from dataclasses import asdict
from typing import Optional

import numpy as np
import torch
import wandb
import wandb.wandb_run

import config1
import config_manager
import loggers


def seed_everything(seed: int):
    """Seeds everything. Cuda seeding and benchmarking configuration excluded for now."""
    random.seed(seed)
    # Set PYTHONHASHSEED environment variable at a fixed value
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_training(config: config_manager.CompleteConfig, wandb_run: Optional[wandb.wandb_run.Run]):
    # Seed
    if config.top_level_config.SEED is not None:
        seed_everything(config.top_level_config.SEED)

    # Get dataloaders
    train_dataloader = config_manager.get_cifar_dataloader(config.data_config, is_train=True)
    val_dataloader = config_manager.get_cifar_dataloader(config.data_config, is_train=False)

    # Get model
    model = config_manager.get_model(config.model_config, is_cifar10=config.data_config.IS_CIFAR10)

    model.to(device=config.top_level_config.DEVICE, dtype=config.top_level_config.DTYPE)
    if config.top_level_config.DEVICE == torch.device("mps"):
        model.compile(backend="aot_eager")
    else:
        model.compile()

    # Get logger
    logger = loggers.Logger(wandb_run)

    optimizer = config_manager.get_optimizer(config.optimizer_config, model)

    lr_scheduler = config_manager.get_lr_scheduler(optimizer, config.lr_scheduler_config)

    # Get trainer
    trainer = config_manager.get_trainer(
        config=config.trainer_config,
        model=model,
        dataloader=train_dataloader,
        logger=logger,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=config.top_level_config.DEVICE,
        dtype=config.top_level_config.DTYPE,
    )

    # Get evaluator
    evaluator = config_manager.get_evaluator(
        config=config.evaluator_config,
        model=model,
        dataloader=val_dataloader,
        device=config.top_level_config.DEVICE,
        dtype=config.top_level_config.DTYPE,
        logger=logger,
    )

    # Run Training
    for epoch in range(config.top_level_config.NUM_EPOCHS):
        trainer.train_epoch(epoch)
        evaluator.eval_epoch(epoch)


if __name__ == "__main__":
    config = config1.config1

    with wandb.init(project=config.top_level_config.WANDB_PROJECT, config=asdict(config)) as run:
        run_training(config, run)
