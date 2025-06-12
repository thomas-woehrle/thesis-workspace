from dataclasses import asdict
import argparse

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
    # Create a dedicated generator for the dataloader, which must be on CPU
    dataloader_generator = torch.Generator(device="cpu")

    # Seed
    if run_config.general_config.SEED is not None:
        utils.seed_everything(run_config.general_config.SEED)
        dataloader_generator.manual_seed(run_config.general_config.SEED)

    # Get model
    model = config.get_model(run_config.model_config, is_cifar10=run_config.data_config.IS_CIFAR10)
    model.to(device=run_config.general_config.DEVICE)

    # Get optimizer
    optimizer = config.get_optimizer(run_config.optimizer_config, model)

    # Get lr scheduler
    lr_scheduler = config.get_lr_scheduler(run_config.lr_scheduler_config, optimizer)

    # Resume from checkpoint if path is provided
    if run_config.general_config.RESUME_FROM_CKPT_PATH:
        train_step, dataloader_rng_state = utils.load_checkpoint(
            model, optimizer, lr_scheduler, run_config.general_config.RESUME_FROM_CKPT_PATH
        )
        if dataloader_rng_state is not None:
            dataloader_generator.set_state(dataloader_rng_state)
    else:
        # train_step is 1-indexed, because that makes the results more interpretable
        train_step = 1

    # Get dataloaders
    train_dataloader = config.get_cifar_dataloader(
        run_config.data_config,
        is_train=True,
        num_train_steps=run_config.general_config.NUM_TRAIN_STEPS,
        generator=dataloader_generator,
    )
    val_dataloader = config.get_cifar_dataloader(run_config.data_config, is_train=False)

    # Get logger
    logger = loggers.Logger(wand_run)

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
        use_torch_compile=run_config.general_config.USE_TORCH_COMPILE,
        mp_dtype=run_config.general_config.MP_DTYPE,
    )

    # Get evaluator
    evaluator = config.get_evaluator(
        config=run_config.evaluator_config,
        model=model,
        dataloader=val_dataloader,
        logger=logger,
    )

    # Compile model
    if run_config.general_config.USE_TORCH_COMPILE and not isinstance(
        optimizer, config.optimizers.EvolutionaryOptimizer
    ):
        # In the evolutionary case, the actual speed-improving compilation happens in the trainer.
        # Otherwise, we compile the model here.
        if run_config.general_config.DEVICE == torch.device("mps"):
            model.compile(backend="aot_eager")
        else:
            model.compile()

    # Run Training
    interval_num = 1
    while train_step <= run_config.general_config.NUM_TRAIN_STEPS:
        num_steps = min(
            run_config.general_config.TRAIN_INTERVAL_LENGTH,
            # +1 because train_step is 1-indexed
            run_config.general_config.NUM_TRAIN_STEPS - train_step + 1,
        )
        trainer.train(train_step, num_steps=num_steps)
        train_step += run_config.general_config.TRAIN_INTERVAL_LENGTH
        evaluator.eval(train_step - 1)

        # Checkpoint
        if (
            run_config.general_config.CKPT_EVERY_NTH_INTERVAL is not None
            and interval_num % run_config.general_config.CKPT_EVERY_NTH_INTERVAL == 0
        ):
            ckpt_path = f"{run_config.general_config.CKPT_PATH}/{wand_run.id}/ckpt_interval_{interval_num}.pt"
            utils.save_checkpoint(
                model, optimizer, lr_scheduler, train_step, ckpt_path, dataloader_generator
            )

        interval_num += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ResNet Training. \n"
        "To start a new run, provide a path to a config file. \n"
        "To resume a run, use the --resume flag with the wandb run ID."
    )
    parser.add_argument(
        "config_file_path",
        nargs="?",
        default=None,
        help="Path to the config file (for new runs).",
    )
    parser.add_argument("--resume", type=str, help="Wandb <project_name>/<run_id> to resume from.")

    args = parser.parse_args()

    if args.config_file_path and args.resume:
        parser.error("argument config_file_path not allowed with argument --resume")

    if not args.config_file_path and not args.resume:
        parser.error("either a config_file_path or --resume is required")

    if args.resume:
        project_name, run_id = args.resume.split("/")
        run_config, wandb_run = utils.resume_from_wandb(run_id, project_name)
    else:
        # The case where args.config_file_path is not None is handled by the check above
        run_config = utils.load_config_from_file(args.config_file_path)
        wandb_run = wandb.init(
            project=run_config.general_config.WANDB_PROJECT,
            config=asdict(run_config),
            mode="online" if run_config.general_config.WANDB_PROJECT is not None else "offline",
        )

    with wandb_run:
        run_training(run_config, wandb_run)
