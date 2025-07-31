import logging
import time
from typing import Any, Mapping

import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback, Checkpoint
from tabulate import tabulate

logger = logging.getLogger(__name__)

class TrainingTimeEstimator(Callback):
    def __init__(
        self,
        num_test_steps: int,
        num_warmup_steps: int = 2,
        enable_checkpointing: bool = False
    ) -> None:
        super().__init__()

        assert num_warmup_steps >= 0
        assert num_warmup_steps < num_test_steps
        
        self.num_test_steps = num_test_steps
        self.num_warmup_steps = num_warmup_steps
        self.enable_checkpointing = enable_checkpointing

        self.start_time = 0
        self.end_time = 0

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        trainer.limit_val_batches = 0
        trainer.accumulate_grad_batches = 1

        if not self.enable_checkpointing:
            trainer.callbacks = [c for c in trainer.callbacks if not isinstance(c, Checkpoint)]

    def print_estimated_training_time(self) -> None:
        seconds = self.end_time - self.start_time
        steps_per_second = (self.num_test_steps - self.num_warmup_steps) / seconds
        estimated_seconds = self.num_total_steps / steps_per_second

        for unit_seconds, unit_name in [
            (60 * 60 * 24, 'days'),
            (60 * 60, 'hours'),
            (60, 'minutes'),
            (1, 'seconds')
        ]:
            if estimated_seconds >= unit_seconds:
                estimated_training_time = f'{estimated_seconds / unit_seconds:.2f} {unit_name}'
                break

        s = tabulate(
            [
                ['Running Time', f'{seconds:.2f} seconds'],
                ['Steps per second', f'{steps_per_second:.2f} steps'],
                ['Estimated training time', estimated_training_time]
            ],
            tablefmt='fancy_grid'
        )
        print(s)

    def on_train_batch_start(self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int) -> None:
        if trainer.global_step == self.num_warmup_steps:
            self.start_time = time.time()

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: torch.Tensor | Mapping[str, Any] | None,
        batch: Any,
        batch_idx: int
    ) -> None:
        if trainer.global_step == self.num_test_steps:
            self.end_time = time.time()
            trainer.should_stop = True

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.num_total_steps = trainer.estimated_stepping_batches

        if trainer.is_global_zero:
            self.print_estimated_training_time()
