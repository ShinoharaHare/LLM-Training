import logging
from functools import partial
from typing import Any, Dict, List

import lightning as L
import torch
from lightning.fabric.plugins import ClusterEnvironment
from lightning.fabric.utilities.types import _PATH
from lightning.pytorch.accelerators import Accelerator
from lightning.pytorch.plugins import Precision
from lightning.pytorch.strategies.deepspeed import \
    DeepSpeedStrategy as _DeepSpeedStrategy
from lightning.pytorch.utilities.types import STEP_OUTPUT


class DeepSpeedStrategy(_DeepSpeedStrategy):
    def __init__(
        self,
        accelerator: Accelerator | None = None,
        zero_optimization: bool = True,
        stage: int = 2,
        remote_device: str | None = None,
        offload_optimizer: bool = False,
        offload_parameters: bool = False,
        offload_params_device: str = 'cpu',
        nvme_path: str = '/local_nvme',
        params_buffer_count: int = 5,
        params_buffer_size: int = 100000000,
        max_in_cpu: int = 1000000000,
        offload_optimizer_device: str = 'cpu',
        optimizer_buffer_count: int = 4,
        block_size: int = 1048576,
        queue_depth: int = 8,
        single_submit: bool = False,
        overlap_events: bool = True,
        thread_count: int = 1,
        pin_memory: bool = False,
        sub_group_size: int = 1000000000000,
        contiguous_gradients: bool = True,
        overlap_comm: bool = True,
        allgather_partitions: bool = True,
        reduce_scatter: bool = True,
        allgather_bucket_size: int = 200000000,
        reduce_bucket_size: int = 200000000,
        zero_allow_untested_optimizer: bool = True,
        logging_batch_size_per_gpu: str | int = 'auto',
        config: _PATH | Dict[str, Any] | None = None,
        logging_level: int | str = logging.WARN,
        parallel_devices: List[torch.device] | None = None,
        cluster_environment: ClusterEnvironment | None = None,
        loss_scale: float = 0,
        initial_scale_power: int = 16,
        loss_scale_window: int = 1000,
        hysteresis: int = 2,
        min_loss_scale: int = 1,
        partition_activations: bool = False,
        cpu_checkpointing: bool = False,
        contiguous_memory_optimization: bool = False,
        synchronize_checkpoint_boundary: bool = False,
        load_full_weights: bool = False,
        precision_plugin: Precision | None = None,
        process_group_backend: str | None = None,
        exclude_frozen_parameters: bool = True,
        raise_error_at_min_scale: bool | None = None,
        zero3_leaf_modules: list[type] | None = None,
        stage3_max_live_parameters: int | float = 1e9,
        stage3_max_reuse_distance: int | float = 1e9,
        stage3_prefetch_bucket_size: int | float = 5e8,
        stage3_param_persistence_threshold: int | float = 1e6,
        zero_hpz_partition_size: int = 1,
        zero_quantized_weights: bool = False,
        zero_quantized_gradients: bool = False
    ):
        if isinstance(logging_level, str):
            logging_level = getattr(logging, logging_level.upper())

        self.exclude_frozen_parameters = exclude_frozen_parameters
        self.raise_error_at_min_scale = raise_error_at_min_scale
        self.zero3_leaf_modules = zero3_leaf_modules
        self.stage3_max_live_parameters = stage3_max_live_parameters
        self.stage3_max_reuse_distance = stage3_max_reuse_distance
        self.stage3_prefetch_bucket_size = stage3_prefetch_bucket_size
        self.stage3_param_persistence_threshold = stage3_param_persistence_threshold
        self.zero_hpz_partition_size = zero_hpz_partition_size
        self.zero_quantized_weights = zero_quantized_weights
        self.zero_quantized_gradients = zero_quantized_gradients

        super().__init__(accelerator, zero_optimization, stage, remote_device, offload_optimizer, offload_parameters, offload_params_device, nvme_path, params_buffer_count, params_buffer_size, max_in_cpu, offload_optimizer_device, optimizer_buffer_count, block_size, queue_depth, single_submit, overlap_events, thread_count, pin_memory, sub_group_size, contiguous_gradients, overlap_comm, allgather_partitions, reduce_scatter, allgather_bucket_size, reduce_bucket_size, zero_allow_untested_optimizer, logging_batch_size_per_gpu, config, logging_level, parallel_devices, cluster_environment, loss_scale, initial_scale_power, loss_scale_window, hysteresis, min_loss_scale, partition_activations, cpu_checkpointing, contiguous_memory_optimization, synchronize_checkpoint_boundary, load_full_weights, precision_plugin, process_group_backend)

    @property
    def is_fp16(self) -> bool:
        return self.precision_plugin.precision.startswith('16')

    def _create_default_config(self, *args, **kwargs) -> dict[str, Any]:
        kwargs.setdefault('stage3_max_live_parameters', self.stage3_max_live_parameters)
        kwargs.setdefault('stage3_max_reuse_distance', self.stage3_max_reuse_distance)
        kwargs.setdefault('stage3_prefetch_bucket_size', self.stage3_prefetch_bucket_size)
        kwargs.setdefault('stage3_param_persistence_threshold', self.stage3_param_persistence_threshold)
        kwargs.setdefault('zero_hpz_partition_size', self.zero_hpz_partition_size)
        kwargs.setdefault('zero_quantized_weights', self.zero_quantized_weights)
        kwargs.setdefault('zero_quantized_gradients', self.zero_quantized_gradients)
        return super()._create_default_config(*args, **kwargs)

    def _set_raise_error_at_min_scale(self):
        optimizer = getattr(self.deepspeed_engine, 'optimizer', None)
        loss_scaler = getattr(optimizer, 'loss_scaler', None)
        if self.raise_error_at_min_scale is not None and loss_scaler is not None:
            loss_scaler.raise_error_at_min_scale = self.raise_error_at_min_scale
    
    def _convert_metrics(self):
        from torchmetrics import Metric

        for m in self.model.modules():
            if isinstance(m, Metric):
                m.to(self.root_device)
                m.set_dtype(m.dtype)
    
    def init_deepspeed(self) -> None:
        import deepspeed  # type: ignore
        
        if self.zero3_leaf_modules:
            deepspeed.utils.set_z3_leaf_modules(self.model, self.zero3_leaf_modules)
        super().init_deepspeed()

    def setup(self, trainer: L.Trainer) -> None:
        super().setup(trainer)

        self._set_raise_error_at_min_scale()
        self._convert_metrics()

    def _maybe_add_skipped_steps_to_progress_bar(self):
        if not self.is_fp16:
            return
        
        progress_bar_metrics = self.lightning_module.trainer.progress_bar_metrics
        progress_bar_metrics['skipped_steps'] = self.deepspeed_engine.skipped_steps

    def training_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        self.lightning_module._grad_norm = None
        output = super().training_step(*args, **kwargs)
        self._maybe_add_skipped_steps_to_progress_bar()
        return output
    
    def optimizer_step(self, optimizer, closure, model = None, **kwargs):
        output = super().optimizer_step(optimizer, closure, model, **kwargs)
        self.lightning_module._grad_norm = self.deepspeed_engine.get_global_grad_norm()
        return output

    def save_checkpoint(self, checkpoint: dict, filepath: _PATH, storage_options: Any | None = None) -> None:
        save_checkpoint = self.deepspeed_engine.save_checkpoint
        self.deepspeed_engine.save_checkpoint = partial(
            save_checkpoint,
            exclude_frozen_parameters=self.exclude_frozen_parameters
        )
        super().save_checkpoint(checkpoint, filepath, storage_options)
        self.deepspeed_engine.save_checkpoint = save_checkpoint
