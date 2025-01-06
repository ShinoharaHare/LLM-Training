import inspect
import logging
import re
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, ContextManager, ParamSpec, TypeVar

import safetensors.torch
import torch
import torch.distributed
import torch.utils.checkpoint
from lightning import LightningModule, Trainer
from lightning.pytorch.core.optimizer import LightningOptimizer
from lightning.pytorch.strategies import Strategy
from torch import nn
from torch.distributed.tensor import DTensor, distribute_tensor
from torch.optim.optimizer import Optimizer
from tqdm.auto import tqdm

from llm_training.lightning import DeepSpeedStrategy, FSDP2Strategy
from llm_training.models.base_model.base_model import BaseModel
from llm_training.models.utils import init_empty_weights
from llm_training.utils.context_managers import ContextManagers

from .base_lm_config import BaseLightningModuleConfig

P = ParamSpec('P')
R = TypeVar('R')

logger = logging.getLogger(__name__)

class BaseLightningModule(LightningModule):
    def __init__(self, config: BaseLightningModuleConfig) -> None:
        super().__init__()

        self.config = config
        self._grad_norm = None
        self.configure_model = self._wrap_configure_model(self.configure_model)
        
    @property
    def has_pre_trained_weights(self) -> bool:
        return self.config.pre_trained_weights is not None

    @property
    def strategy(self) -> Strategy | None:
        return None if self._trainer is None else self.trainer.strategy
    
    @property
    def is_using_ds_z3(self) -> bool:
        return (
            isinstance(self.strategy, DeepSpeedStrategy)
            and self.strategy.zero_stage_3
        )

    @property
    def is_using_fsdp(self) -> bool:
        return isinstance(self.strategy, FSDP2Strategy)
    
    @property
    def is_loading_from_checkpoint(self) -> bool:
        if self._trainer is None:
            return False
        
        if self.trainer.ckpt_path is not None:
            return True

        current_frame = inspect.currentframe().f_back
        while current_frame is not None:
            f_locals = current_frame.f_locals
            current_frame = current_frame.f_back

            if isinstance(f_locals.get('self', None), Trainer) and 'ckpt_path' in f_locals:
                return f_locals['ckpt_path'] is not None

    @property
    def should_load_pre_trained_weights(self) -> bool:
        return (
            self.has_pre_trained_weights
            and self.config.load_weights
            and not self.is_loading_from_checkpoint
        )
    
    @property
    def should_initialize_weights(self) -> bool:
        return (
            self.config.init_weights
            and not self.is_loading_from_checkpoint
            and not self.config.load_weights
        )
    
    @property
    def should_sync_weights(self) -> bool:
        return self.should_load_pre_trained_weights or self.should_initialize_weights

    @property
    def grad_norm(self) -> torch.Tensor | float | None:
        return self._grad_norm

    def get_pre_trained_weights(self) -> dict[str, torch.Tensor]:
        return safetensors.torch.load_file(self.config.pre_trained_weights)

    @staticmethod
    @contextmanager
    def shutdown_ds_init_context():
        import deepspeed  # type: ignore
        deepspeed.zero.partition_parameters.shutdown_init_context()
        yield
        deepspeed.zero.partition_parameters.restore_init_context()

    def get_pre_trained_weights_context(self) -> ContextManagers:
        context_managers = []
        if isinstance(self.strategy, DeepSpeedStrategy):
            context_managers.append(self.shutdown_ds_init_context())
        return ContextManagers(context_managers)

    def _get_weight_loading_progress_bar(
        self,
        model: nn.Module | None = None,
        name: str | None = None
    ) -> tqdm:
        model = model or self
        desc = 'Loading weights'
        if name is not None:
            desc = f'{desc} ({name})'
        
        disable = self.global_rank != 0
        disable |= not self.should_initialize_weights and not self.should_load_pre_trained_weights

        return tqdm(
            desc=desc,
            total=sum(1 for _ in model.parameters()),
            disable=disable
        )

    def _ds_z3_load_state_dict(self, state_dict: dict[str, torch.Tensor] | None):
        import deepspeed  # type: ignore

        with self._get_weight_loading_progress_bar() as progress_bar:
            for n, p in self.named_parameters():
                with deepspeed.zero.GatheredParameters([p], modifier_rank=0):
                    if self.global_rank == 0:
                        p.data.copy_(state_dict[n])
                
                progress_bar.set_postfix_str(n)
                progress_bar.update()

    def _fsdp2_load_state_dict(self, state_dict: dict[str, torch.Tensor] | None):
        with self._get_weight_loading_progress_bar() as progress_bar:
            for n, p in self.named_parameters():
                if isinstance(p, DTensor):
                    w = torch.empty(
                        p.shape,
                        dtype=p.dtype,
                        device=self.strategy.root_device
                    )

                    if self.global_rank == 0:
                        w.data.copy_(state_dict[n])

                    w = distribute_tensor(
                        w,
                        p.device_mesh,
                        p.placements
                    )
                    p.data.copy_(w)
                else:
                    if self.global_rank == 0:
                        p.data.copy_(state_dict[n])
                    
                    torch.distributed.broadcast(p.data, src=0)

                progress_bar.set_postfix_str(n)
                progress_bar.update()
    
    def load_pre_trained_weights(self) -> None:
        with self.get_pre_trained_weights_context():
            state_dict = self.get_pre_trained_weights() if self.global_rank == 0 else None
        
        if self.is_using_ds_z3:
            self._ds_z3_load_state_dict(state_dict)
        elif self.is_using_fsdp:
            self._fsdp2_load_state_dict(state_dict)
        elif self._trainer is not None and self._trainer.num_devices > 1:
            with self._get_weight_loading_progress_bar() as progress_bar:
                if self.global_rank == 0:
                    self.load_state_dict(state_dict)
                
                for n, p in self.named_parameters():
                    torch.distributed.broadcast(p.data, src=0)
                    progress_bar.set_postfix_str(n)
                    progress_bar.update()
        else:
            self.load_state_dict(state_dict)

    def configure_model_context(self) -> ContextManager:
        if self.is_using_fsdp:
            return init_empty_weights(include_buffers=False)
        return BaseModel.init_weights_context(self.should_initialize_weights)

    def on_fsdp_parallelize_model(self, **kwargs) -> None:
        raise NotImplementedError()

    @contextmanager
    def fsdp_parallelize_context(self):
        named_weights = [(False, n, p) for n, p in self.named_parameters(remove_duplicate=False)]
        named_weights += [(True, n, b) for n, b in self.named_buffers(remove_duplicate=False)]

        memo = {}
        shared_weights = []
        for is_buffer, n, w in named_weights:
            if w in memo:
                shared_weights.append((is_buffer, n, memo[w]))
            else:
                memo[w] = n

        yield

        for m in self.modules():
            if (
                any(t.is_meta for t in m.parameters(recurse=False))
                or any(t.is_meta for t in m.buffers(recurse=False))
            ):
                m.to_empty(device=self.strategy.root_device, recurse=False)

        for is_buffer, k, v in shared_weights:
            mn, _, wn = k.rpartition('.')
            m = self.get_submodule(mn)
            if is_buffer:
                m.register_buffer(wn, self.get_buffer(v))
            else:
                m.register_parameter(wn, self.get_parameter(v))

    def on_after_configure_model(self) -> None:
        if self.config.frozen_modules is not None:
            for n, m in self.named_modules():
                for fm in self.config.frozen_modules:
                    if re.search(rf'{fm}', n) is not None:
                        if self.global_rank == 0:
                            logger.info(f'Freeze `{n}`')
                        m.eval().requires_grad_(False)
                        break

        if isinstance(self.strategy, FSDP2Strategy):
            with self.fsdp_parallelize_context():
                self.on_fsdp_parallelize_model(
                    dp_mesh=self.strategy.dp_mesh,
                    tp_mesh=self.strategy.tp_mesh,
                    reshard_after_forward=self.strategy.reshard_after_forward,
                    mp_policy=self.strategy.mp_policy,
                    offload_policy=self.strategy.offload_policy
                )
        
        if self.should_load_pre_trained_weights:
            self.load_pre_trained_weights()
    
    def _wrap_configure_model(self, configure_model: Callable[P, R]) -> Callable[P, R]:
        @wraps(configure_model)
        def wrapped_configure_model():
            with self.configure_model_context():
                configure_model()
            
            self.on_after_configure_model()

            if self.strategy is not None:
                self.strategy.barrier()

        return wrapped_configure_model
    
    def configure_optimizers(self):
        assert self.config.optim is not None

        optimizer = self.config.optim.optimizer_class(self.parameters(), **self.config.optim.optimizer_kwargs)

        lr_scheduler_class = self.config.optim.lr_scheduler_class
        lr_scheduler_parameters = inspect.signature(lr_scheduler_class).parameters
        lr_scheduler_kwargs = {}
        if 'num_total_steps' in lr_scheduler_parameters:
            lr_scheduler_kwargs['num_total_steps'] = self.trainer.estimated_stepping_batches
        lr_scheduler_kwargs |= self.config.optim.lr_scheduler_kwargs
        lr_scheduler = self.config.optim.lr_scheduler_class(optimizer, **lr_scheduler_kwargs)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': lr_scheduler,
                'interval': 'step'
            }
        }
    
    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: Optimizer | LightningOptimizer,
        optimizer_closure: Callable[[], Any] | None = None
    ) -> None:
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        
        if self.config.log_grad_norm and self.grad_norm is not None:
            self.log('Gradient Norm', self.grad_norm)

    @property
    def required_keys(self) -> set[str]:
        model = self.get_model()
        for n, m in self.named_children():
            if m is model:
                prefix = n + '.'
                break
        else:
            raise Exception("Failed to infer prefix")

        state_dict = model.state_dict(prefix=prefix)
        return set(state_dict.keys())
