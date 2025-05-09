from config_manager import *

config1 = CompleteConfig(
    top_level_config=TopLevelConfig(
        WANDB_PROJECT="thesis_baseline_testruns",
        DEVICE=torch.device("cpu"),
        DTYPE=torch.float32,
        SEED=None,
        NUM_EPOCHS=10,
    ),
    data_config=DataConfig(
        BATCH_SIZE=16, NUM_WORKERS=0, USE_EVERY_NTH=1000, IS_CIFAR10=False, USE_IMG_TRANSFORMS=True
    ),
    model_config=ModelConfig(
        MODEL_SLUG="small_resnet20", USE_INSTANCE_NORM=False, BN_TRACK_RUNNING_STATS=True
    ),
    trainer_config=TrainerConfig(TRAINER_SLUG="backprop_trainer"),
    optimizer_config=OptimizerConfig(OPTIMIZER_SLUG="sgd", LR=0.001),
    lr_scheduler_config=LRSchedulerConfig(LR_SCHEDULER_SLUG=None),
    evaluator_config=EvaluatorConfig(DO_LOG_MODELS=False),
)
