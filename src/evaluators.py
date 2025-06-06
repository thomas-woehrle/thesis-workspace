from typing import Callable

import torch
import torch.nn as nn
import tqdm
from torch.utils.data import DataLoader

import loggers


def get_num_topk_correct_preds(batch_logits: torch.Tensor, batch_target: torch.Tensor, topk: int):
    _, batch_topk_indices = torch.topk(batch_logits, topk, dim=-1)
    batch_topk_pred_correct = (batch_topk_indices == batch_target.unsqueeze(-1)).sum(dim=-1) > 0
    return batch_topk_pred_correct.sum(dim=-1)


class Evaluator1:
    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        criterion: Callable,
        do_log_models: bool,
        logger: loggers.Logger,
    ):
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.do_log_models = do_log_models
        self.logger = logger

    def eval_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch
        x, y = (
            x.to(
                device=next(self.model.parameters()).device,
            ),
            y.to(device=next(self.model.parameters()).device),
        )

        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)

        # For accuracy statistics
        total_preds = y.shape[0]
        top1_correct_preds = get_num_topk_correct_preds(y_hat, y, 1)
        top5_correct_preds = get_num_topk_correct_preds(y_hat, y, 5)

        return loss, total_preds, top1_correct_preds, top5_correct_preds

    @torch.no_grad()
    def eval(self, current_train_step: int):
        """Validate the current model"""
        self.model.eval()

        losses = torch.zeros(len(self.dataloader))
        total_preds = torch.zeros(len(self.dataloader))
        top1_correct_preds = torch.zeros(len(self.dataloader))
        top5_correct_preds = torch.zeros(len(self.dataloader))

        for batch_idx, batch in enumerate(
            tqdm.tqdm(self.dataloader, leave=False, desc=f"Validation - Step {current_train_step}")
        ):
            (
                batch_loss,
                batch_total_preds,
                batch_top1_correct_preds,
                batch_top5_correct_preds,
            ) = self.eval_step(batch, batch_idx)
            losses[batch_idx] = batch_loss
            total_preds[batch_idx] = batch_total_preds
            top1_correct_preds[batch_idx] = batch_top1_correct_preds
            top5_correct_preds[batch_idx] = batch_top5_correct_preds

        self.logger.log(
            {"val/top1_accuracy": top1_correct_preds.sum().item() / total_preds.sum().item()},
            current_train_step,
        )
        self.logger.log(
            {"val/top5_accuracy": top5_correct_preds.sum().item() / total_preds.sum().item()},
            current_train_step,
        )
        self.logger.log({"val/loss": losses.mean().item()}, current_train_step)
        if self.do_log_models:
            self.logger.log_model(self.model)
