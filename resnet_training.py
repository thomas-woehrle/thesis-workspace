import sys
from dataclasses import asdict

import torch
import wandb
import wandb.wandb_run

import config
import loggers
import utils


def run_training(
    run_config: config.RunConfig,
    wand_run: wandb.wandb_run.Run,
):
    # Seed
    if run_config.general_config.SEED is not None:
        utils.seed_everything(run_config.general_config.SEED)

    # Get dataloaders
    train_dataloader = config.get_cifar_dataloader(
        run_config.data_config,
        is_train=True,
        num_train_steps=run_config.general_config.NUM_TRAIN_STEPS,
    )
    val_dataloader = config.get_cifar_dataloader(run_config.data_config, is_train=False)

    # Get model
    model = config.get_model(run_config.model_config, is_cifar10=run_config.data_config.IS_CIFAR10)
    model.to(device=run_config.general_config.DEVICE, dtype=run_config.general_config.DTYPE)

    if run_config.general_config.DEVICE == torch.device("mps"):
        model.compile(backend="aot_eager")
    else:
        model.compile()

    # Get logger
    logger = loggers.Logger(wand_run)

    # Get optimizer
    optimizer = config.get_optimizer(run_config.optimizer_config, model)

    # Get lr scheduler
    lr_scheduler = config.get_lr_scheduler(run_config.lr_scheduler_config, optimizer)

    # Get trainer
    trainer = config.get_trainer(
        config=run_config.trainer_config,
        model=model,
        dataloader=train_dataloader,
        logger=logger,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        use_instance_norm=run_config.model_config.USE_INSTANCE_NORM,
        bn_track_running_stats=run_config.model_config.BN_TRACK_RUNNING_STATS,
    )

    # Get evaluator
    evaluator = config.get_evaluator(
        config=run_config.evaluator_config,
        model=model,
        dataloader=val_dataloader,
        logger=logger,
    )

    # Run Training
    # train_step is 1-indexed, because that makes the results more interpretable
    train_step = 1
    while train_step <= run_config.general_config.NUM_TRAIN_STEPS:
        num_steps = min(
            run_config.general_config.TRAIN_INTERVAL_LENGTH,
            # +1 because train_step is 1-indexed
            run_config.general_config.NUM_TRAIN_STEPS - train_step + 1,
        )
        trainer.train(train_step, num_steps=num_steps)
        train_step += run_config.general_config.TRAIN_INTERVAL_LENGTH
        evaluator.eval(train_step - 1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise ValueError("Usage: python resnet_training.py <path_to_config_file>")

    config_file_path = sys.argv[1]

    run_config = utils.load_config_from_file(config_file_path)

    with wandb.init(
        project=run_config.general_config.WANDB_PROJECT, config=asdict(run_config)
    ) as wandb_run:
        run_training(run_config, wandb_run)
