from contextlib import contextmanager
from typing import ClassVar, Mapping

import torch
from torch import nn
from torch.optim import Optimizer
from typing_extensions import Self


class MasterWeightsOptimizer(Optimizer):
    _is_subclass: ClassVar[bool] = False
    _subclasses: ClassVar[dict[type[Optimizer], type[Self]]] = {}

    _parameter_mapping: Mapping[nn.Parameter, nn.Parameter]
    _parameters: list[nn.Parameter]

    def __new__(cls, optimizer: Optimizer):
        if cls._is_subclass:
            return object.__new__(cls)

        optimizer_class = type(optimizer)
        if optimizer_class not in cls._subclasses:
            cls._subclasses[optimizer_class] = type(
                optimizer_class.__name__,
                (cls,),
                {'_is_subclass': True}
            )

        return cls._subclasses[optimizer_class](optimizer)

    def __init__(self, optimizer: Optimizer):        
        self._optimizer = optimizer
        self._parameters = [p for g in self._optimizer.param_groups for p in g['params']]
        self._parameter_mapping = {}
        for p in self._parameters:
            mp = p if p.dtype == torch.float else p.detach().float().requires_grad_()
            self._parameter_mapping[p] = mp
            self._parameter_mapping[mp] = p

    @contextmanager
    def _replace_params(self, replace_state_key: bool = True):
        try:
            for group in self._optimizer.param_groups:
                params = group['params']
                for i, p in enumerate(params):
                    params[i] = self._parameter_mapping[p]

            if replace_state_key:
                for w in list(self._optimizer.state.keys()):
                    self._optimizer.state[self._parameter_mapping[w]] = self._optimizer.state.pop(w)
        
            yield
        finally:
            for group in self._optimizer.param_groups:
                params = group['params']
                for i, p in enumerate(params):
                    params[i] = self._parameter_mapping[p]

            if replace_state_key:
                for mw in list(self._optimizer.state.keys()):
                    self._optimizer.state[self._parameter_mapping[mw]] = self._optimizer.state.pop(mw)
    
    def step(self, closure=None):        
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for p in self._parameters:
            if p.grad is None:
                continue
            self._parameter_mapping[p].grad = p.grad.float()

        with self._replace_params():
            self._optimizer.step()

        for p in self._parameters:
            p.data.copy_(self._parameter_mapping[p], non_blocking=True)

        return loss
    
    def zero_grad(self, set_to_none = True):
        self._optimizer.zero_grad(set_to_none)

        with self._replace_params(replace_state_key=False):
            self._optimizer.zero_grad(set_to_none)

    def state_dict(self):
        return self._optimizer.state_dict()
    
    def load_state_dict(self, state_dict):
        with self._replace_params():
            self._optimizer.load_state_dict(state_dict)

    def __getattr__(self, name):
        return getattr(self._optimizer, name)
