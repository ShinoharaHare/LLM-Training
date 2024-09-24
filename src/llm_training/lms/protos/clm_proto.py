from typing import Protocol, overload

import torch
from torch import nn

class CausalLMProto(Protocol):
    def get_input_embeddings(self) -> nn.Embedding: ...
    
    def get_output_embeddings(self) -> nn.Linear: ...

    @overload
    def __call__(
        self,
        *,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        input_embeds: torch.Tensor | None = None
    ) -> torch.Tensor: ...

    @overload
    def __call__(
        self,
        *,
        input_ids: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        input_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
