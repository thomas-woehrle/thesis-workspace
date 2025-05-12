import sys
from dataclasses import asdict

import torch
import wandb
import wandb.wandb_run

import config
import loggers
import utils


def run_training(
    general_config: config.GeneralConfig,
    data_config: config.DataConfig,
    model_config: config.ModelConfig,
    trainer_config: config.TrainerConfig,
    optimizer_config: config.OptimizerConfig,
    lr_scheduler_config: config.LRSchedulerConfig,
    evaluator_config: config.EvaluatorConfig,
    wand_run: wandb.wandb_run.Run,
):
    # Seed
    if general_config.SEED is not None:
        utils.seed_everything(general_config.SEED)

    # Get dataloaders
    train_dataloader = config.get_cifar_dataloader(data_config, is_train=True)
    val_dataloader = config.get_cifar_dataloader(data_config, is_train=False)

    # Get model
    model = config.get_model(model_config, is_cifar10=data_config.IS_CIFAR10)
    model.to(device=general_config.DEVICE, dtype=general_config.DTYPE)

    if general_config.DEVICE == torch.device("mps"):
        model.compile(backend="aot_eager")
    else:
        model.compile()

    # Get logger
    logger = loggers.Logger(wand_run)

    optimizer = config.get_optimizer(optimizer_config, model)

    lr_scheduler = config.get_lr_scheduler(optimizer, lr_scheduler_config)

    # Get trainer
    trainer = config.get_trainer(
        config=trainer_config,
        model=model,
        dataloader=train_dataloader,
        logger=logger,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        use_instance_norm=model_config.USE_INSTANCE_NORM,
        bn_track_running_stats=model_config.BN_TRACK_RUNNING_STATS,
        device=general_config.DEVICE,
        dtype=general_config.DTYPE,
    )

    # Get evaluator
    evaluator = config.get_evaluator(
        config=evaluator_config,
        model=model,
        dataloader=val_dataloader,
        device=general_config.DEVICE,
        dtype=general_config.DTYPE,
        logger=logger,
    )

    # Run Training
    for epoch in range(general_config.NUM_EPOCHS):
        trainer.train_epoch(epoch)
        evaluator.eval_epoch(epoch)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise ValueError("Usage: python resnet_training.py <path_to_config_file>")

    config_file_path = sys.argv[1]

    run_config = utils.load_config_from_file(config_file_path)

    with wandb.init(
        project=run_config.general_config.WANDB_PROJECT, config=asdict(run_config)
    ) as wandb_run:
        run_training(
            general_config=run_config.general_config,
            data_config=run_config.data_config,
            trainer_config=run_config.trainer_config,
            model_config=run_config.model_config,
            optimizer_config=run_config.optimizer_config,
            lr_scheduler_config=run_config.lr_scheduler_config,
            evaluator_config=run_config.evaluator_config,
            wand_run=wandb_run,
        )
