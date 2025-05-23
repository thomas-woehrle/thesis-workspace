import os
import time
from typing import Any, Optional

import torch
import torch.nn as nn
import wandb.wandb_run


class Logger:
    def __init__(self, wandb_run: Optional[wandb.wandb_run.Run]):
        self.wandb_run = wandb_run

    def log(self, data: dict[str, Any], step: Optional[int] = None, commit: Optional[bool] = None):
        """Effectively mirrors wandb run.log API, see https://docs.wandb.ai/ref/python/run/#log"""
        if self.wandb_run is not None:
            self.wandb_run.log(data, step, commit)
        else:
            print(f"Step {step}: {data}")

    def log_model(
        self, model: nn.Module, name: (str | None) = None, aliases: (list[str] | None) = None
    ):
        """Effectively mirrors wandb run.log_model API"""
        # Get current time and create a name from it to prevent name collisions when multiple trainings run at the same time
        current_time_ns = int(time.time())
        path = f"./temp_model_{current_time_ns}"
        torch.save(model.state_dict(), path)
        if self.wandb_run is not None:
            self.wandb_run.log_model(path, name, aliases)
            os.remove(path)
        else:
            raise NotImplementedError()
