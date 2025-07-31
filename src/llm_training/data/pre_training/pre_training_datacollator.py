import math
from typing import Any

import torch

from llm_training.data.base_datacollator import BaseDataCollator
from .pre_training_datamodule_config import PreTrainingDataModuleConfig


class PreTrainingDataCollator(BaseDataCollator):
    config: PreTrainingDataModuleConfig

    @property
    def tokenizer(self):
        return self.config.tokenizer

    def __init__(self, config: PreTrainingDataModuleConfig) -> None:
        super().__init__(config)

        assert 'pad_token' in config.tokenizer.special_tokens_map, '`pad_token` is not specified. Please set it manually.'
    
    def _pad_to_longest(self, x: list[list[int]]) -> list[list[int]]:
        n = self.config.max_length if self.config.pad_to_max_length else max(len(y) for y in batch)

        if self.config.pad_to_multiple_of is not None:
            n = (math.ceil(n / self.config.pad_to_multiple_of)) * self.config.pad_to_multiple_of
        
        for y in x:
            num_paddings = n - len(y)
            paddings = [-1] * num_paddings
            y[:] = paddings + y if self.tokenizer.padding_side == 'left' else y + paddings
        return x

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_ids = [x['input_ids'] for x in batch]
        
        input_ids = self._pad_to_longest(input_ids)
        input_ids = torch.tensor(input_ids)
        padding_mask = input_ids == -1
        input_ids[padding_mask] = self.tokenizer.pad_token_id
        bos_mask = input_ids == self.tokenizer.bos_token_id

        return {
            'input_ids': input_ids,
            'attention_mask': torch.ones_like(input_ids).masked_fill(padding_mask, 0),
            'position_ids': torch.arange(input_ids.size(1)).unsqueeze(0),
            'labels': input_ids.masked_fill(bos_mask | padding_mask, -100)
        }
