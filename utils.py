import os
import random


import numpy as np
import torch
import yaml

import config
from config import RunConfig


def load_config_from_file(file_path: str) -> RunConfig:
    with open(file_path, "r") as f:
        config_dict = yaml.safe_load(f)

    # Create dataclass instances from config dictionary
    general_config_dict = config_dict.get("GENERAL_CONFIG")
    general_config_dict["DEVICE"] = torch.device(general_config_dict.get("DEVICE", "cpu"))
    general_config_dict["DTYPE"] = getattr(torch, general_config_dict.get("DTYPE", "float32"))
    general_config = config.GeneralConfig(**general_config_dict)

    model_config = config.ModelConfig(**config_dict.get("MODEL_CONFIG", {}))
    data_config = config.DataConfig(**config_dict.get("DATA_CONFIG", {}))
    optimizer_config = config.OptimizerConfig(**config_dict.get("OPTIMIZER_CONFIG", {}))
    lr_scheduler_config = config.LRSchedulerConfig(**config_dict.get("LR_SCHEDULER_CONFIG", {}))
    trainer_config = config.TrainerConfig(**config_dict.get("TRAINER_CONFIG", {}))
    evaluator_config = config.EvaluatorConfig(**config_dict.get("EVALUATOR_CONFIG", {}))

    return RunConfig(
        general_config=general_config,
        model_config=model_config,
        data_config=data_config,
        optimizer_config=optimizer_config,
        lr_scheduler_config=lr_scheduler_config,
        trainer_config=trainer_config,
        evaluator_config=evaluator_config,
    )


def seed_everything(seed: int):
    """Seeds everything. Cuda seeding and benchmarking configuration excluded for now."""
    random.seed(seed)
    # Set PYTHONHASHSEED environment variable at a fixed value
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
