import torch

from .metric import Metric


class Loss(Metric):
    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False
    
    loss: torch.Tensor
    count: torch.Tensor

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.add_state('loss', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0.0), dist_reduce_fx='sum')

    def update(self, loss: torch.Tensor):
        self.loss += loss
        self.count += 1
    
    def compute(self):
        return self.loss / self.count
