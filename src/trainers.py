from abc import ABC, abstractmethod
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import wandb
from torch.utils.data import DataLoader
from torch._functorch.apis import vmap
from scipy.stats import mannwhitneyu

import loggers
import optimizers
import utils


class Trainer(ABC):
    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: optim.Optimizer,
        lr_scheduler: Optional[optim.lr_scheduler.LRScheduler],
        criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        mp_dtype: torch.dtype,
        logger: loggers.Logger,
    ):
        self.model = model
        self.dataloader = dataloader
        self.data_iterator = iter(self.dataloader)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.criterion = criterion
        self.mp_dtype = mp_dtype
        self.logger = logger

    @abstractmethod
    def train_step(self, x: torch.Tensor, y: torch.Tensor, step: int) -> torch.Tensor | float:
        """Can return float or tensor. Tensor should be preferred if train_epoch uses it as tensor anyway,
        which will often be the case. Reason is that the .item() conversion blocks the GPU for a very short piece of time."""
        raise NotImplementedError()

    def train(self, start_train_step: int, num_steps: int):
        self.model.train()

        losses = torch.zeros(num_steps)
        for i in tqdm.tqdm(
            range(num_steps),
            leave=False,
            desc=f"Training - Steps {start_train_step} - {start_train_step + num_steps - 1}",
        ):
            x, y = next(self.data_iterator)
            x, y = (
                x.to(
                    device=next(self.model.parameters()).device,
                    non_blocking=True,
                ),
                y.to(device=next(self.model.parameters()).device, non_blocking=True),
            )
            losses[i] = self.train_step(x, y, start_train_step + i)

            if self.lr_scheduler:
                self.lr_scheduler.step()

        # Logging
        self.logger.log({"train/loss": losses.mean().item()}, start_train_step + num_steps - 1)

        if self.lr_scheduler:
            self.logger.log(
                {"info/lr": self.lr_scheduler.get_last_lr()[0]}, start_train_step + num_steps - 1
            )


class BackpropagationTrainer(Trainer):
    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor | float:
        self.optimizer.zero_grad()

        with torch.autocast(
            device_type=next(self.model.parameters()).device.type,
            dtype=self.mp_dtype,
            enabled=self.mp_dtype != torch.float32,
        ):
            y_hat = self.model(x)
            loss = self.criterion(y_hat, y)

        loss.backward()
        self.optimizer.step()

        return loss

    def train(self, start_train_step: int, num_steps: int):
        super().train(start_train_step, num_steps)


class EvolutionaryTrainer(Trainer):
    # TODO change to EvolutionaryOptimizer
    optimizer: optimizers.EvolutionaryOptimizer

    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: optimizers.EvolutionaryOptimizer,
        lr_scheduler: Optional[optim.lr_scheduler.LRScheduler],
        criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        use_parallel_forward_pass: bool,
        bn_track_running_stats: bool,
        use_instance_norm: bool,
        mp_dtype: torch.dtype,
        logger: loggers.Logger,
        use_torch_compile: bool,
        do_adaptation_sampling_every_nth_step: Optional[int],
        adaptation_sampling_p_value_threshold: Optional[float],
    ):
        super().__init__(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            criterion=criterion,
            mp_dtype=mp_dtype,
            logger=logger,
        )
        self.use_parallel_forward_pass = use_parallel_forward_pass
        self.parallel_forward_pass_fn = utils.get_parallel_forward_pass_fn(self.model)
        if use_torch_compile:
            self.parallel_forward_pass_fn = torch.compile(self.parallel_forward_pass_fn)

        self.do_adaptation_sampling_every_nth_step = do_adaptation_sampling_every_nth_step
        self.adaptation_sampling_p_value_threshold = adaptation_sampling_p_value_threshold

        self.bn_track_running_stats = bn_track_running_stats
        self.use_instance_norm = use_instance_norm
        self.batched_criterion = vmap(self.criterion, in_dims=(0, None))
        self.popsize = self.optimizer.popsize
        self.batched_named_buffers = {
            # manually moved to the correct dtype, because autocast doesn't realiably do it in this case
            n: torch.stack([b] * self.popsize, dim=0).to(dtype=self.mp_dtype)
            for n, b in self.model.named_buffers()
        }

    def _get_batched_named_params(
        self, batched_flat_params: torch.Tensor, popsize: int
    ) -> dict[str, torch.Tensor]:
        batched_flat_params_split = batched_flat_params.split(
            [p.numel() for p in self.model.parameters()], dim=1
        )

        batched_named_params = {
            n: batched_flat_p.view(popsize, *p.shape)
            for (n, p), batched_flat_p in zip(
                self.model.named_parameters(), batched_flat_params_split
            )
        }

        return batched_named_params

    def _evaluate_adaptation_sampling(
        self, x: torch.Tensor, y: torch.Tensor, losses_current_sigma: torch.Tensor, step: int
    ):
        if not isinstance(self.optimizer, optimizers.SNESOptimizer):
            return

        params_adapted, _, _ = self.optimizer.get_new_generation(from_adaptation_sampled_sigma=True)
        losses_adapted = self._get_losses(params_adapted, x, y)

        p_value = mannwhitneyu(
            losses_adapted.cpu().numpy(),
            losses_current_sigma.cpu().numpy(),
            alternative="less",
        )[1]

        self.logger.log({"info/as_p_value": p_value}, step)

        if p_value < self.adaptation_sampling_p_value_threshold:
            # aggressive update was good, increase lr
            self.optimizer.sigma_param_group["lr"] *= self.optimizer.adaptation_sampling_c_prime
        else:
            # aggressive update was bad, decrease lr
            self.optimizer.sigma_param_group["lr"] /= self.optimizer.adaptation_sampling_c_prime

    def _get_losses(
        self, batched_flat_params: torch.Tensor, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        mutated_batched_named_params = self._get_batched_named_params(
            batched_flat_params, self.popsize
        )

        popsize = batched_flat_params.shape[0]

        with torch.autocast(
            device_type=next(self.model.parameters()).device.type,
            dtype=self.mp_dtype,
            enabled=self.mp_dtype != torch.float32,
        ):
            if self.use_parallel_forward_pass:
                y_hat = self.parallel_forward_pass_fn(
                    (mutated_batched_named_params, self.batched_named_buffers), x
                )
                # Convert to full precision for loss computation to ensure numerical stability
                losses = self.batched_criterion(y_hat.float(), y)

            else:
                losses = torch.zeros(
                    popsize,
                    device=batched_flat_params.device,
                )

                for i in range(popsize):
                    nn.utils.vector_to_parameters(batched_flat_params[i], self.model.parameters())
                    y_hat = self.model(x)
                    # Convert to full precision for loss computation to ensure numerical stability
                    losses[i] = self.criterion(y_hat.float(), y)

        return losses

    def train_step(self, x: torch.Tensor, y: torch.Tensor, step: int) -> torch.Tensor | float:
        mutated_batched_flat_params, mutations, zs = self.optimizer.get_new_generation()
        losses = self._get_losses(mutated_batched_flat_params, x, y)

        if (
            self.do_adaptation_sampling_every_nth_step is not None
            and step % self.do_adaptation_sampling_every_nth_step == 0
        ):
            self._evaluate_adaptation_sampling(x, y, losses, step)

        self.optimizer.step(losses, mutations, zs)
        return losses.mean()

    @torch.no_grad()
    def train(self, start_train_step: int, num_steps: int):
        super().train(start_train_step, num_steps)

        if (
            self.use_parallel_forward_pass
            and not self.use_instance_norm
            and self.bn_track_running_stats
        ):
            named_buffers = {n: b[0] for n, b in self.batched_named_buffers.items()}
            self.model.load_state_dict(named_buffers, strict=False)

        if not isinstance(self.optimizer, optimizers.BlockDiagonalNESOptimizer):
            sigma_mean = (
                self.optimizer.sigma.mean()
                if isinstance(self.optimizer.sigma, torch.Tensor)
                else self.optimizer.sigma
            )
            self.logger.log({"info/sigma_mean": sigma_mean}, start_train_step + num_steps - 1)
            if isinstance(self.optimizer.sigma, torch.Tensor):
                self.logger.log(
                    {"info/sigma distribution": wandb.Histogram(self.optimizer.sigma.tolist())},
                    start_train_step + num_steps - 1,
                )

            if isinstance(self.optimizer, optimizers.SNESOptimizer):
                self.logger.log(
                    {
                        "info/lr_mu": self.optimizer.mu_param_group["lr"],
                        "info/lr_sigma": self.optimizer.sigma_param_group["lr"],
                    },
                    start_train_step + num_steps - 1,
                )
