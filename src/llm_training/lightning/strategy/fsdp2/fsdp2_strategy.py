import datetime
import logging
import shutil
from contextlib import contextmanager, nullcontext
from datetime import timedelta
from pathlib import Path
from typing import (Any, Dict, Generator, List, Literal, Mapping, Optional,
                    Union)

import lightning.pytorch as pl
import torch
from lightning.fabric.strategies.model_parallel import (
    _distributed_checkpoint_save, _is_sharded_checkpoint, _load_checkpoint,
    _setup_device_mesh)
from lightning.fabric.utilities.distributed import (
    _distributed_is_initialized, _get_default_process_group_backend_for_device,
    _init_dist_connection, _sync_ddp_if_available)
from lightning.fabric.utilities.distributed import group as _group
from lightning.fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_4
from lightning.fabric.utilities.init import _materialize_distributed_module
from lightning.fabric.utilities.load import _METADATA_FILENAME
from lightning.fabric.utilities.optimizer import _optimizers_to_device
from lightning.fabric.utilities.seed import reset_seed
from lightning.fabric.utilities.types import _PATH, ReduceOp
from lightning.pytorch.core.optimizer import LightningOptimizer
from lightning.pytorch.plugins.precision import Precision
from lightning.pytorch.strategies.launchers.subprocess_script import \
    _SubprocessScriptLauncher
from lightning.pytorch.strategies.parallel import ParallelStrategy
from lightning.pytorch.strategies.strategy import TBroadcast
from lightning.pytorch.trainer.states import TrainerFn
from lightning.pytorch.utilities.model_helpers import is_overridden
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from lightning_utilities.core.rank_zero import \
    rank_zero_only as utils_rank_zero_only
from torch.distributed._composable.fsdp import (MixedPrecisionPolicy,
                                                OffloadPolicy)
from torch.distributed.device_mesh import DeviceMesh
from torch.optim import Optimizer
from typing_extensions import override

from llm_training.optim.master_weight_wrapper import MasterWeightsOptimizer

from .fsdp2_precision import FSDP2Precision

logger = logging.getLogger(__name__)

class FSDP2Strategy(ParallelStrategy):
    def __init__(
        self,
        data_parallel_size: int | Literal["auto"] = 'auto',
        tensor_parallel_size: int | Literal["auto"] = 1,
        save_distributed_checkpoint: bool = True,
        process_group_backend: Optional[str] = None,
        timeout: Optional[timedelta] = datetime.timedelta(minutes=30),
        reshard_after_forward: bool | int = True,
        mp_policy: MixedPrecisionPolicy | None = None,
        offload_policy: OffloadPolicy = OffloadPolicy(),
        use_master_weights: bool = True
    ) -> None:
        super().__init__()

        if not _TORCH_GREATER_EQUAL_2_4:
            raise ImportError(f"{type(self).__name__} requires PyTorch 2.4 or higher.")
        
        self._data_parallel_size = data_parallel_size
        self._tensor_parallel_size = tensor_parallel_size
        self._save_distributed_checkpoint = save_distributed_checkpoint
        self._process_group_backend: Optional[str] = process_group_backend
        self._timeout: Optional[timedelta] = timeout
        self._device_mesh: Optional[DeviceMesh] = None
        self.num_nodes = 1

        self._reshard_after_forward = reshard_after_forward
        self._mp_policy = mp_policy
        self._offload_policy = offload_policy
        
        self._use_master_weights = use_master_weights

    @property
    @override
    def precision_plugin(self) -> FSDP2Precision:
        return self._precision_plugin
    
    @precision_plugin.setter
    @override
    def precision_plugin(self, precision_plugin: Precision) -> None:
        if precision_plugin is None:
            self._precision_plugin = None
        else:
            self._precision_plugin = FSDP2Precision(precision_plugin.precision)

    @property
    def reshard_after_forward(self) -> bool | int:
        return self._reshard_after_forward

    @property
    def mp_policy(self) -> MixedPrecisionPolicy:
        return self._mp_policy or self.precision_plugin.mp_policy
    
    @property
    def offload_policy(self) -> OffloadPolicy:
        return self._offload_policy

    @property
    def device_mesh(self) -> DeviceMesh:
        if self._device_mesh is None:
            raise RuntimeError("Accessing the device mesh before processes have initialized is not allowed.")
        return self._device_mesh
    
    @property
    def dp_mesh(self) -> DeviceMesh:
        return self._device_mesh['data_parallel']

    @property
    def tp_mesh(self) -> DeviceMesh:
        return self._device_mesh['tensor_parallel']
    
    @property
    def dp_size(self) -> int:
        return self.dp_mesh.size()

    @property
    def tp_size(self) -> int:
        return self.tp_mesh.size()

    @property
    def dp_rank(self) -> int:
        return self.dp_mesh.get_local_rank()
    
    @property
    def tp_rank(self) -> int:
        return self.tp_mesh.get_local_rank()

    @property
    @override
    def root_device(self) -> torch.device:
        assert self.parallel_devices is not None
        return self.parallel_devices[self.local_rank]

    @property
    def num_processes(self) -> int:
        return len(self.parallel_devices) if self.parallel_devices is not None else 0
    
    @property
    def is_distributed(self) -> bool:
        return True

    @property
    @override
    def distributed_sampler_kwargs(self) -> dict[str, Any]:
        assert self.device_mesh is not None
        return {'num_replicas': self.dp_size, 'rank': self.dp_rank}

    @property
    def process_group_backend(self) -> Optional[str]:
        return self._process_group_backend

    @property
    @override
    def restore_checkpoint_after_setup(self) -> bool:
        return True

    @property
    @override
    def lightning_restore_optimizer(self) -> bool:
        return False

    @override
    def _configure_launcher(self) -> None:
        assert self.cluster_environment is not None
        if not self.cluster_environment.creates_processes_externally:
            self._launcher = _SubprocessScriptLauncher(self.cluster_environment, self.num_processes, self.num_nodes)

    @override
    def setup_environment(self) -> None:
        super().setup_environment()
        
        self._setup_distributed()

        if self._data_parallel_size == 'auto' and self._tensor_parallel_size == 'auto':
            self._data_parallel_size = self.num_nodes
            self._tensor_parallel_size = self.num_processes
        elif self._data_parallel_size == 'auto' and self._tensor_parallel_size != 'auto':
            assert self.world_size % self._tensor_parallel_size == 0
            self._data_parallel_size = self.world_size // self._tensor_parallel_size
        elif self._data_parallel_size != 'auto' and self._tensor_parallel_size == 'auto':
            assert self.world_size % self._data_parallel_size == 0
            self._tensor_parallel_size = self.world_size // self._data_parallel_size
        else:
            assert self.world_size == self._data_parallel_size * self._tensor_parallel_size

        if self.is_global_zero:
            logger.info(f'Data Parallel Size: {self._data_parallel_size}')
            logger.info(f'Tensor Parallel Size: {self._tensor_parallel_size}')
        
        self._device_mesh = _setup_device_mesh(
            self._data_parallel_size, self._tensor_parallel_size, self.world_size, self.root_device
        )

        # Users can access device mesh in `LightningModule.configure_model()`
        assert self.lightning_module is not None
        self.lightning_module._device_mesh = self._device_mesh

    @override
    def setup(self, trainer: "pl.Trainer") -> None:
        from torch.distributed.fsdp import FullyShardedDataParallel

        assert self.model is not None
        assert self.accelerator is not None
        self.accelerator.setup(trainer)

        if not is_overridden("configure_model", self.lightning_module):
            raise TypeError(
                f"When using the {type(self).__name__}, you are required to override the `configure_model()` hook in"
                f" the LightningModule and apply parallelization there."
            )
        
        if any(isinstance(mod, FullyShardedDataParallel) for mod in self.model.modules()):
            raise TypeError(
                "Found modules that are wrapped with `torch.distributed.fsdp.FullyShardedDataParallel`."
                f" The `{self.__class__.__name__}` only supports the new FSDP2 APIs in PyTorch >= 2.4."
            )

        _materialize_distributed_module(self.model, self.root_device)

        self.model = self.precision_plugin.convert_module(self.model)
        self.model_to_device()  # move all remaining layers if any left on CPU.

        self.barrier()

        if trainer.state.fn == TrainerFn.FITTING:
            self.setup_optimizers(trainer)
        
        self.setup_precision_plugin()
        
        if trainer.state.fn == TrainerFn.FITTING:
            _optimizers_to_device(self.optimizers, self.root_device)

    @override
    def setup_optimizers(self, trainer: "pl.Trainer") -> None:
        # If we're setting up for evaluation after fitting, we need to discard the optimizers
        # since we're rewrapping the model, otherwise optimizer param references are no longer valid
        # and subsequent checkpoint saving can fail
        self._reset_optimizers_and_schedulers()

        super().setup_optimizers(trainer)

        if self.mp_policy.param_dtype in (torch.half, torch.bfloat16) and self._use_master_weights:
            self.optimizers = [MasterWeightsOptimizer(optimizer) for optimizer in self.optimizers]

    @override
    def model_to_device(self) -> None:
        assert self.model is not None
        self.model.to(self.root_device)

    @contextmanager
    @override
    def tensor_init_context(self, empty_init: Optional[bool] = None) -> Generator[None, None, None]:
        # Materializaton happens in `setup()`
        empty_init_context = torch.device("meta") if empty_init else nullcontext()
        with empty_init_context, self.precision_plugin.tensor_init_context():
            yield

    @override
    def barrier(self, name: Optional[str] = None) -> None:
        if not _distributed_is_initialized():
            return
        if torch.distributed.get_backend() == "nccl":
            torch.distributed.barrier(device_ids=self._determine_device_ids())
        else:
            torch.distributed.barrier()

    @override
    def broadcast(self, obj: TBroadcast, src: int = 0) -> TBroadcast:
        if not _distributed_is_initialized():
            return obj

        obj = [obj]
        torch.distributed.broadcast_object_list(obj, src, group=_group.WORLD)
        return obj[0]

    @override
    def reduce(
        self,
        tensor: Union[torch.Tensor, Any],
        group: Optional[Any] = None,
        reduce_op: Optional[Union[ReduceOp, str]] = "mean" # type: ignore
    ) -> torch.Tensor:
        if isinstance(tensor, torch.Tensor):
            return _sync_ddp_if_available(tensor, group, reduce_op=reduce_op)
        return tensor

    def _determine_device_ids(self) -> List[int]:
        return [self.root_device.index]
    
    def training_step(self, *args, **kwargs):
        self.lightning_module._grad_norm = None
        return super().training_step(*args, **kwargs)
    
    def optimizer_step(self, optimizer, closure, model = None, **kwargs):
        output = super().optimizer_step(optimizer, closure, model, **kwargs)
        self.lightning_module._grad_norm = self.precision_plugin._grad_norm
        return output

    @override
    def teardown(self) -> None:
        assert self.cluster_environment is not None
        assert self.accelerator is not None
        self.cluster_environment.teardown()
        self.precision_plugin.teardown()
        self.accelerator.teardown()

    @override
    def lightning_module_state_dict(self) -> Dict[str, Any]:
        """Collects the state dict of the model.

        Only returns a non-empty state dict on rank 0 if ``save_distributed_checkpoint=False``.

        """
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions, get_model_state_dict)

        state_dict_options = StateDictOptions(full_state_dict=not self._save_distributed_checkpoint, cpu_offload=True)
        assert self.model is not None
        return get_model_state_dict(self.model, options=state_dict_options)

    @override
    def load_model_state_dict(self, checkpoint: Mapping[str, Any], strict: bool = True) -> None:
        # Override to do nothing, the strategy already loaded the states in `load_checkpoint()`
        pass

    @override
    def optimizer_state(self, optimizer: Optimizer) -> Dict[str, Any]:
        """Collects the state of the given optimizer.

        Only returns a non-empty state dict on rank 0 if ``save_distributed_checkpoint=False``.

        """
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions, get_optimizer_state_dict)
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import OptimStateKeyType

        state_dict_options = StateDictOptions(full_state_dict=(not self._save_distributed_checkpoint), cpu_offload=True)
        if isinstance(optimizer, LightningOptimizer):
            optimizer = optimizer._optimizer

        assert self.model is not None

        state_dict = get_optimizer_state_dict(self.model, optimizer, options=state_dict_options)
        if not self._save_distributed_checkpoint and self.global_rank == 0:
            # Store the optimizer state dict in standard format
            state_dict = FSDP.rekey_optim_state_dict(state_dict, OptimStateKeyType.PARAM_ID, self.model)
        return state_dict

    @override
    def load_optimizer_state_dict(self, checkpoint: Mapping[str, Any]) -> None:
        # Override to do nothing, the strategy already loaded the states in `load_checkpoint()`
        pass

    @override
    def save_checkpoint(
        self, checkpoint: Dict[str, Any], filepath: _PATH, storage_options: Optional[Any] = None
    ) -> None:
        if storage_options is not None:
            raise TypeError(
                f"`{type(self).__name__}.save_checkpoint(..., storage_options=...)` is not supported because"
                f" `{type(self).__name__}` does not use the `CheckpointIO`."
            )
        # broadcast the path from rank 0 to ensure all the checkpoints are saved to a common path
        path = Path(self.broadcast(filepath))
        if path.is_dir() and not self._save_distributed_checkpoint and not _is_sharded_checkpoint(path):
            raise IsADirectoryError(f"The checkpoint path exists and is a directory: {path}")

        if self._save_distributed_checkpoint:
            if path.is_file():
                path.unlink()
            path.mkdir(parents=True, exist_ok=True)

            converted_state = {"state_dict": checkpoint.pop("state_dict")}
            converted_state.update({
                f"optimizer_{idx}": optim_state
                for idx, optim_state in enumerate(checkpoint.pop("optimizer_states", []))
            })
            _distributed_checkpoint_save(converted_state, path)

            if self.global_rank == 0:
                torch.save(checkpoint, path / _METADATA_FILENAME)
        else:
            if _is_sharded_checkpoint(path):
                shutil.rmtree(path)
            return super().save_checkpoint(checkpoint=checkpoint, filepath=path)

    @override
    def load_checkpoint(self, checkpoint_path: _PATH) -> Dict[str, Any]:
        # broadcast the path from rank 0 to ensure all the states are loaded from a common path
        path = Path(self.broadcast(checkpoint_path))
        state = {
            "state_dict": self.model,
            **{f"optimizer_{idx}": optimizer for idx, optimizer in enumerate(self.optimizers)},
        }
        assert self.lightning_module is not None
        return _load_checkpoint(
            path=path,
            state=state,
            strict=self.lightning_module.strict_loading,
            optimizer_states_from_list=True,
        )

    def _setup_distributed(self) -> None:
        super().setup_environment()
        reset_seed()
        self.set_world_ranks()
        self._process_group_backend = self._get_process_group_backend()
        assert self.cluster_environment is not None
        _init_dist_connection(self.cluster_environment, self._process_group_backend, timeout=self._timeout)

    def _get_process_group_backend(self) -> str:
        return self._process_group_backend or _get_default_process_group_backend_for_device(self.root_device)

    def set_world_ranks(self) -> None:
        if self.cluster_environment is not None:
            self.cluster_environment.set_global_rank(self.node_rank * self.num_processes + self.local_rank)
            self.cluster_environment.set_world_size(self.num_nodes * self.num_processes)
        # `LightningEnvironment.set_global_rank` will do this too, but we cannot rely on that implementation detail
        # additionally, for some implementations, the setter is a no-op, so it's safer to access the getter
        rank_zero_only.rank = utils_rank_zero_only.rank = self.global_rank

    @contextmanager
    def block_backward_sync(self) -> Generator:
        from torch.distributed._composable.fsdp import FSDPModule
        
        for m in self.lightning_module.modules():
            if isinstance(m, FSDPModule):
                m.set_requires_gradient_sync(False, recurse=False)
        
        yield

        for m in self.lightning_module.modules():
            if isinstance(m, FSDPModule):
                m.set_requires_gradient_sync(True, recurse=False)
