# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
import warnings
from typing import List

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class LinearWarmupCosineAnnealingLR(_LRScheduler):
    """Sets the learning rate of each parameter group to follow a linear warmup schedule between
    warmup_start_lr and base_lr followed by a cosine annealing schedule between base_lr and
    eta_min."""

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_epochs (int): Maximum number of iterations for linear warmup
            max_epochs (int): Maximum number of iterations
            warmup_start_lr (float): Learning rate to start the linear warmup. Default: 0.
            eta_min (float): Minimum learning rate. Default: 0.
            last_epoch (int): The index of last epoch. Default: -1.
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:  # type: ignore
        """Compute learning rate using chainable form of the scheduler."""
        if not self._get_lr_called_within_step:  # type: ignore
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch == self.warmup_epochs:
            return self.base_lrs
        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)
        if self.last_epoch < self.warmup_epochs:
            return [
                group["lr"] + (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        if (self.last_epoch - 1 - self.max_epochs) % (
            2 * (self.max_epochs - self.warmup_epochs)
        ) == 0:
            return [
                group["lr"]
                + (base_lr - self.eta_min)
                * (1 - math.cos(math.pi / (self.max_epochs - self.warmup_epochs)))
                / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        return [
            (
                1
                + math.cos(
                    math.pi
                    * (self.last_epoch - self.warmup_epochs)
                    / (self.max_epochs - self.warmup_epochs)
                )
            )
            / (
                1
                + math.cos(
                    math.pi
                    * (self.last_epoch - self.warmup_epochs - 1)
                    / (self.max_epochs - self.warmup_epochs)
                )
            )
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self) -> List[float]:
        """Called when epoch is passed as a param to the `step` function of the scheduler."""
        if self.last_epoch < self.warmup_epochs:
            return [
                self.warmup_start_lr
                + self.last_epoch
                * (base_lr - self.warmup_start_lr)
                / max(1, self.warmup_epochs - 1)
                for base_lr in self.base_lrs
            ]

        return [
            self.eta_min
            + 0.5
            * (base_lr - self.eta_min)
            * (
                1
                + math.cos(
                    math.pi
                    * (self.last_epoch - self.warmup_epochs)
                    / (self.max_epochs - self.warmup_epochs)
                )
            )
            for base_lr in self.base_lrs
        ]
    
def configure_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    n_steps: int,
    num_devices: int,
    max_epochs: int,
    warmup_epochs: int,
    warmup_start_lr: float,
    eta_min: float,
    num_nodes: int,
    **kwargs,  # accept any params
):
    """`n_steps` is `len(train_sampler)` at the call site (train.py), and a PyTorch
    `DistributedSampler.__len__` is ALREADY the per-rank shard size
    (`ceil(len(dataset) / world_size)`), not the global dataset size. This function used to divide
    that already-per-rank count by `num_devices * num_nodes` a second time, which shrank
    `n_steps` by another factor of world size and made `warmup_epochs` (`warmup_epochs_arg *
    n_steps`) both node-count-dependent and, on every shape this project has run
    (`warmup_epochs_arg=10 > EPOCHS<=3` always), LARGER than `max_epochs`
    (`max_epochs_arg * n_steps`) -- inverting the cosine-annealing denominator
    `max_epochs - warmup_epochs` to negative. `LinearWarmupCosineAnnealingLR` is a CHAINABLE
    scheduler (each step's LR is a delta from the current value, not an absolute formula), and its
    cosine branches divide by `(1 + cos(...))`, which can approach zero and blow the LR up when the
    period is wrong; because the update is relative, one bad step compounds into every step after
    it. At 4 Corona nodes this bug put the warmup/cosine boundary at step ~110 of a 200-step run,
    matching where a real run's `comm`/`fetch` broke down (`STATUS-STORMER-4NODE-BALANCE-
    2026-09-08.md`) and its loss diverged (`STATUS-STORMER-DIVERGENCE-DIAGNOSIS-2026-09-08.md`).
    FIXED (2026-09-08): `n_steps` is used as passed, with no second division, so `warmup_epochs`
    stays the intended `warmup_epochs_arg` real training epochs' worth of per-rank steps --
    thousands, on this subset, at any node count. This does NOT make `warmup_epochs < max_epochs`
    true (that inequality is `warmup_epochs_arg(10) < EPOCHS`, a separate, still-hardcoded
    relationship this fix does not touch): what it restores is that `warmup_epochs` is now large
    enough that no run of 200 steps or fewer, at any node count this project uses, ever crosses
    into the cosine-annealing branch at all, which is the property that actually kept the schedule
    safe. See `liveflow_shm/tests/test_scheduler_warmup.py` for the regression test and the
    informational check that documents `warmup_epochs > max_epochs` remaining true.
    """
    lr_scheduler = LinearWarmupCosineAnnealingLR(
        optimizer=optimizer,
        warmup_epochs=warmup_epochs * n_steps,
        max_epochs=max_epochs * n_steps,
        warmup_start_lr=warmup_start_lr,
        eta_min=eta_min,
    )
    return lr_scheduler
