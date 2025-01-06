from typing import Any, Callable, ContextManager, Literal

import torch
from lightning import LightningModule
from lightning.fabric.plugins.precision.amp import _optimizer_handles_unscaling
from lightning.fabric.plugins.precision.utils import (_convert_fp_tensor,
                                                      _DtypeContextManager)
from lightning.fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_4
from lightning.fabric.utilities.types import Optimizable
from lightning.pytorch.plugins.precision.precision import Precision
from lightning.pytorch.utilities import GradClipAlgorithmType
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning_utilities import apply_to_collection
from torch.distributed._composable.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor import DTensor
from torch.optim import LBFGS, Optimizer
from typing_extensions import get_args, override

_PRECISION_INPUT = Literal["32-true", "16-true", "bf16-true", "16-mixed", "bf16-mixed"]
_HALF_PRECISIONS = ('16-true', 'bf16-true')
_MIXED_PRECISIONS = ('16-mixed', 'bf16-mixed')


class FSDP2Precision(Precision):
    def __init__(
        self,
        precision: _PRECISION_INPUT,
        device: str = 'cuda',
        scaler: "torch.amp.GradScaler | None" = None,
    ) -> None:
        supported_precision = get_args(_PRECISION_INPUT)
        if precision not in supported_precision:
            raise ValueError(
                f"`precision={precision!r})` is not supported in FSDP."
                f" `precision` must be one of: {supported_precision}."
            )

        if scaler is None and self.precision == "16-mixed":
            scaler = torch.amp.GradScaler(device=device) if _TORCH_GREATER_EQUAL_2_4 else torch.cuda.amp.GradScaler()
        
        self.precision = precision
        self.device = device
        self.scaler = scaler

        precision_to_type = {
            'bf16-mixed': torch.float32,
            '16-mixed': torch.float32,
            'bf16-true': torch.bfloat16,
            '16-true': torch.float16,
            '32-true': torch.float32
        }
        self._desired_input_dtype = precision_to_type[self.precision]
        self._grad_norm = None

    @property
    def mp_policy(self) -> "MixedPrecisionPolicy":
        if self.precision == '16-mixed':
            param_dtype = torch.float
            reduce_dtype = torch.half
            output_dtype = torch.half
            cast_forward_inputs = False
        elif self.precision == 'bf16-mixed':
            param_dtype = torch.float
            reduce_dtype = torch.bfloat16
            output_dtype = torch.bfloat16
            cast_forward_inputs = False
        elif self.precision == '16-true':
            param_dtype = torch.half
            reduce_dtype = torch.half
            output_dtype = torch.half
            cast_forward_inputs = False
        elif self.precision == 'bf16-true':
            param_dtype = torch.bfloat16
            reduce_dtype = torch.bfloat16
            output_dtype = torch.bfloat16
            cast_forward_inputs = False
        elif self.precision == '32-true':
            param_dtype = torch.float
            reduce_dtype = torch.float
            output_dtype = torch.float
            cast_forward_inputs = False
        else:
            raise MisconfigurationException(f"Was unable to infer precision type, received {self.precision!r}.")

        return MixedPrecisionPolicy(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            output_dtype=output_dtype,
            cast_forward_inputs=cast_forward_inputs
        )
    
    @override
    def convert_module(self, module: torch.nn.Module) -> torch.nn.Module:
        if self.precision in _HALF_PRECISIONS:
            return module.to(dtype=self._desired_input_dtype)
        return module

    @override
    def tensor_init_context(self) -> ContextManager:
        return _DtypeContextManager(self._desired_input_dtype)

    @override
    def module_init_context(self) -> ContextManager:
        return _DtypeContextManager(self.mp_policy.param_dtype or torch.float32)

    @override
    def forward_context(self) -> ContextManager:
        if self.precision in _HALF_PRECISIONS:
            return _DtypeContextManager(self._desired_input_dtype)

        if self.precision in _MIXED_PRECISIONS:
            return torch.autocast(
                self.device,
                dtype=torch.bfloat16 if self.precision == 'bf16-mixed' else torch.half
            )
        
        return super().forward_context()

    @override
    def pre_backward(self, tensor: torch.Tensor, module: LightningModule) -> torch.Tensor:  # type: ignore[override]
        if self.scaler is not None:
            tensor = self.scaler.scale(tensor)
        return super().pre_backward(tensor, module)
    
    @override
    def convert_input(self, data: Any) -> Any:
        return apply_to_collection(data, function=_convert_fp_tensor, dtype=torch.Tensor, dst_type=self._desired_input_dtype)

    @override
    def optimizer_step(  # type: ignore[override]
        self,
        optimizer: Optimizable,
        model: LightningModule,
        closure: Callable[[], Any],
        **kwargs: Any,
    ) -> Any:
        if self.scaler is None:
            # skip scaler logic, as bfloat16 does not require scaler
            return super().optimizer_step(optimizer, model=model, closure=closure, **kwargs)
        
        if isinstance(optimizer, LBFGS):
            raise MisconfigurationException("AMP and the LBFGS optimizer are not compatible.")
        closure_result = closure()

        # If backward was skipped in automatic optimization (return None), unscaling is not needed
        skip_unscaling = closure_result is None and model.automatic_optimization

        if not _optimizer_handles_unscaling(optimizer) and not skip_unscaling:
            # Unscaling needs to be performed here in case we are going to apply gradient clipping.
            # Optimizers that perform unscaling in their `.step()` method are not supported (e.g., fused Adam).
            # Note: `unscale` happens after the closure is executed, but before the `on_before_optimizer_step` hook.
            self.scaler.unscale_(optimizer)  # type: ignore[arg-type]

        self._after_closure(model, optimizer)

        # in manual optimization, the closure does not return a value
        if not skip_unscaling:
            # note: the scaler will skip the `optimizer.step` if nonfinite gradients are found
            step_output = self.scaler.step(optimizer, **kwargs)  # type: ignore[arg-type]
            self.scaler.update()
            return step_output
        
        return closure_result
    
    @override
    def clip_grad_by_norm(self, optimizer, clip_val):
        parameters = self.main_params(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, clip_val)
        self._grad_norm = grad_norm.full_tensor() if isinstance(grad_norm, DTensor) else grad_norm        
    
    @override
    def clip_gradients(
        self,
        optimizer: Optimizer,
        clip_val: int | float = 0.0,
        gradient_clip_algorithm: GradClipAlgorithmType = GradClipAlgorithmType.NORM,
    ) -> None:
        if self.precision in _MIXED_PRECISIONS and clip_val > 0 and _optimizer_handles_unscaling(optimizer):
            raise RuntimeError(
                f"The current optimizer, {type(optimizer).__qualname__}, does not allow for gradient clipping"
                " because it performs unscaling of gradients internally. HINT: Are you using a 'fused' optimizer?"
            )
        super().clip_gradients(optimizer=optimizer, clip_val=clip_val, gradient_clip_algorithm=gradient_clip_algorithm)

    @override
    def state_dict(self) -> dict[str, Any]:
        if self.scaler is not None:
            return self.scaler.state_dict()
        return {}

    @override
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if self.scaler is not None:
            self.scaler.load_state_dict(state_dict)
