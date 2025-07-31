import logging
import os
from typing import Literal

import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks.callback import Callback
from triton.runtime.cache import default_cache_dir

logger = logging.getLogger(__name__)


class ExtraConfig(Callback):
    def __init__(
        self,
        float32_matmul_precision: Literal["medium", "high", "highest"] | None = None,
        logging_level: int | str = logging.INFO,
        env: dict[str, str] | None = None
    ) -> None:
        super().__init__()

        self.float32_matmul_precision = float32_matmul_precision
        self.logging_level = logging_level
        self.env = env

        self._configure_float32_matmul_precision()
        self._configure_logging_level()
        self._configure_environment()
        
    def _configure_float32_matmul_precision(self) -> None:
        if self.float32_matmul_precision is not None:
            torch.set_float32_matmul_precision(self.float32_matmul_precision)

    def _configure_logging_level(self) -> None:
        if isinstance(self.logging_level, str):
            logging_level = getattr(logging, self.logging_level.upper())
        else:
            logging_level = self.logging_level

        logging.getLogger('llm_training').setLevel(logging_level)
        logging.getLogger('lightning').setLevel(logging_level)

    def _configure_environment(self) -> None:
        if self.env is not None:
            os.environ.update(self.env)

    def _configure_triton_cache_dir(self, rank: int) -> None:
        if not os.getenv('TRITON_CACHE_DIR', '').strip():
            os.environ['TRITON_CACHE_DIR'] = os.path.join(default_cache_dir(), 'llm_training', f'rank_{rank}')

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str):
        self._configure_triton_cache_dir(trainer.global_rank)
