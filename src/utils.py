import os
import random
from typing import Iterable, Optional


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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
    general_config_dict["MP_DTYPE"] = getattr(torch, general_config_dict.get("MP_DTYPE", "float32"))
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
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    lr_scheduler: Optional[optim.lr_scheduler.LRScheduler],
    train_step: int,
    ckpt_path: str,
):
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": lr_scheduler.state_dict() if lr_scheduler else None,
        "train_step": train_step,
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "random_rng_state": random.getstate(),
    }
    torch.save(checkpoint, ckpt_path)


def load_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    lr_scheduler: Optional[optim.lr_scheduler.LRScheduler],
    ckpt_path: str,
):
    checkpoint = torch.load(
        ckpt_path, map_location=next(model.parameters()).device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if lr_scheduler and checkpoint["lr_scheduler_state_dict"]:
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

    torch.set_rng_state(checkpoint["torch_rng_state"].to(device="cpu", dtype=torch.uint8))
    np.random.set_state(checkpoint["numpy_rng_state"])
    random.setstate(checkpoint["random_rng_state"])

    return checkpoint["train_step"]


def get_parallel_forward_pass_fn(
    model: nn.Module,
):
    def forward_pass(parameter_and_buffer_dicts, x):
        return functional_call(model, (parameter_and_buffer_dicts), (x,))

    batched_forward_pass = vmap(forward_pass, in_dims=(0, None))
    return batched_forward_pass


def get_not_supported_message(kind: str, not_supported_slug: str):
    return f"{kind} with slug {not_supported_slug} is not supported..."


def get_non_evolutionary_optimizer(
    slug: str,
    parameters: Iterable[torch.Tensor],
    lr: float,
    momentum: float,
    weight_decay: float,
) -> optim.Optimizer:
    if slug == "sgd":
        return optim.SGD(
            parameters,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    elif slug == "adam":
        return optim.Adam(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
        )
    elif slug == "adamw":
        return optim.AdamW(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(get_not_supported_message("Non-Evolutionary Optimizer", slug))
