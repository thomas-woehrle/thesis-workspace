import os
import random


import numpy as np
import torch
import torch.nn as nn
import yaml
from torch._functorch.functional_call import functional_call
from torch._functorch.apis import vmap

import config


def load_config_from_file(file_path: str) -> "config.RunConfig":
    with open(file_path, "r") as f:
        config_dict = yaml.safe_load(f)

    # Create dataclass instances from config dictionary
    general_config_dict = config_dict.get("GENERAL_CONFIG", {})
    general_config_dict["DEVICE"] = torch.device(general_config_dict.get("DEVICE", "cpu"))
    general_config_dict["DTYPE"] = getattr(torch, general_config_dict.get("DTYPE", "float32"))
    general_config = config.GeneralConfig(**general_config_dict)

    model_config = config.ModelConfig(**config_dict.get("MODEL_CONFIG", {}))
    data_config = config.DataConfig(**config_dict.get("DATA_CONFIG", {}))
    optimizer_config = config.OptimizerConfig(**config_dict.get("OPTIMIZER_CONFIG", {}))
    trainer_config = config.TrainerConfig(**config_dict.get("TRAINER_CONFIG", {}))
    lr_scheduler_config = config.LRSchedulerConfig(**config_dict.get("LR_SCHEDULER_CONFIG", {}))
    evaluator_config = config.EvaluatorConfig(**config_dict.get("EVALUATOR_CONFIG", {}))

    return config.RunConfig(
        general_config=general_config,
        model_config=model_config,
        data_config=data_config,
        trainer_config=trainer_config,
        optimizer_config=optimizer_config,
        lr_scheduler_config=lr_scheduler_config,
        evaluator_config=evaluator_config,
    )


def seed_everything(seed: int):
    """Seeds everything. Cuda seeding and benchmarking configuration excluded for now."""
    random.seed(seed)
    # Set PYTHONHASHSEED environment variable at a fixed value
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parallel_forward_pass(
    model: nn.Module,
    batched_parameter_and_buffer_dicts: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
    x: torch.Tensor,
):
    def forward_pass(parameter_and_buffer_dicts):
        return functional_call(model, (parameter_and_buffer_dicts), (x,))

    batched_forward_pass = vmap(forward_pass)
    return batched_forward_pass(batched_parameter_and_buffer_dicts)


def get_parallel_forward_pass_fn(
    model: nn.Module,
):
    def forward_pass(parameter_and_buffer_dicts, x):
        return functional_call(model, (parameter_and_buffer_dicts), (x,))

    batched_forward_pass = vmap(forward_pass, in_dims=(0, None))
    return batched_forward_pass


def get_not_supported_message(kind: str, not_supported_slug: str):
    return f"{kind} with slug {not_supported_slug} is not supported..."
