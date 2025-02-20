import logging
import os
from typing import Any, Callable

from lightning.pytorch import LightningDataModule, LightningModule
from lightning.pytorch import Trainer as _Trainer
from lightning.pytorch.cli import ArgsType, LightningArgumentParser
from lightning.pytorch.cli import LightningCLI as _LightningCLI
from lightning.pytorch.cli import SaveConfigCallback as _SaveConfigCallback

from llm_training.lightning import (ExtraConfig, OutputRedirection,
                                    SaveConfigCallback, TQDMProgressBar)

from .trainer import Trainer


class LightningCLI(_LightningCLI):
    def __init__(
        self,
        model_class: type[LightningModule] | Callable[..., LightningModule] | None = None,
        datamodule_class: type[LightningDataModule] | Callable[..., LightningDataModule] | None = None,
        save_config_callback: type[_SaveConfigCallback] | None = None,
        save_config_kwargs: dict[str, Any] | None = None,
        trainer_class: type[_Trainer] | Callable[..., _Trainer] = Trainer,
        trainer_defaults: dict[str, Any] | None = None,
        seed_everything_default: bool | int = True,
        parser_kwargs: dict[str, Any] | dict[str, dict[str, Any]] | None = None,
        subclass_mode_model: bool = False,
        subclass_mode_data: bool = False,
        args: ArgsType = None,
        run: bool = True,
        auto_configure_optimizers: bool = True
    ) -> None:
        save_config_callback = SaveConfigCallback if save_config_callback is None else save_config_callback
        default_save_config_kwargs = {
            'overwrite': True,
            'save_to_log_dir': False
        }
        save_config_kwargs = save_config_kwargs or {}
        save_config_kwargs = default_save_config_kwargs | save_config_kwargs

        default_parser_kwargs = {
            'parser_mode': 'omegaconf'
        }
        parser_kwargs = parser_kwargs or {}
        parser_kwargs = default_parser_kwargs | parser_kwargs

        super().__init__(
            model_class=model_class,
            datamodule_class=datamodule_class,
            save_config_callback=save_config_callback,
            save_config_kwargs=save_config_kwargs,
            trainer_class=trainer_class,
            trainer_defaults=trainer_defaults,
            seed_everything_default=seed_everything_default,
            parser_kwargs=parser_kwargs,
            subclass_mode_model=subclass_mode_model,
            subclass_mode_data=subclass_mode_data,
            args=args,
            run=run,
            auto_configure_optimizers=auto_configure_optimizers
        )

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_argument('--float32-matmul-precision', type=str | None, choices=['medium', 'high', 'highest'], default=None)
        parser.add_argument('--logging-level', type=str | int, default=logging.INFO)
        parser.add_lightning_class_args(OutputRedirection, 'output_redirection')
        parser.add_lightning_class_args(TQDMProgressBar, 'tqdm_progress')

    def _instantiate_extra_config(self) -> ExtraConfig:
        return ExtraConfig(
            float32_matmul_precision=self._get(self.config, 'float32_matmul_precision'),
            logging_level=self._get(self.config, 'logging_level')
        )

    def _instantiate_trainer(self, config, callbacks):
        callbacks.insert(0, self._instantiate_extra_config())

        if int(os.getenv('SLURM_NTASKS', '0')) == 1:
            del os.environ['SLURM_JOB_ID']
            del os.environ['SLURM_NTASKS']
        
        return super()._instantiate_trainer(config, callbacks)
