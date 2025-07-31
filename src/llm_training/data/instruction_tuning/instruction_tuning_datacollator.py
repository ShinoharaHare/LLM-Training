import math
from typing import Any, TypeVar

import torch

from llm_training.data.base_datacollator import BaseDataCollator

from .instruction_tuning_datamodule_config import (
    PackingMethod, InstructionTuningDataModuleConfig)

T = TypeVar('T')

class InstructionTuningDataCollator(BaseDataCollator):
    config: InstructionTuningDataModuleConfig

    def __init__(self, config: InstructionTuningDataModuleConfig):
        super().__init__(config)

        assert 'pad_token' in config.tokenizer.special_tokens_map, '`pad_token` is not specified. Please set it manually.'
    
    def _pad_to_longest(self, batch: list[list[T]], padding_value: T) -> list[list[T]]:
        n = self.config.max_length if self.config.pad_to_max_length else max(len(y) for y in batch)

        if self.config.pad_to_multiple_of is not None:
            n = (math.ceil(n / self.config.pad_to_multiple_of)) * self.config.pad_to_multiple_of
        
        new_batch = []
        for x in batch:
            num_paddings = n - len(x)
            paddings = [padding_value] * num_paddings
            x = paddings + x if self.config.tokenizer.padding_side == 'left' else x + paddings
            new_batch.append(x)
        
        return new_batch

    def __call__(self, batch: list[dict[str, Any]]):
        batch_input_ids = []
        batch_attention_mask = []
        batch_position_ids = []
        batch_labels = []
        
        for x in batch:
            input_ids = x['input_ids']
            labels = x['labels']
            n = len(input_ids)

            if self.config.packing_method == PackingMethod.NO_PACKING:
                position_ids = list(range(n))
                attention_mask = [1] * n
            elif self.config.packing_method == PackingMethod.GROUP_BY_LENGTH:
                position_ids = list(range(n))
                attention_mask = x['attention_mask']

            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_position_ids.append(position_ids)
            batch_labels.append(labels)

        batch_input_ids = self._pad_to_longest(batch_input_ids, self.config.tokenizer.pad_token_id)
        batch_attention_mask = self._pad_to_longest(batch_attention_mask, 0)
        batch_position_ids = self._pad_to_longest(batch_position_ids, 0)
        batch_labels = self._pad_to_longest(batch_labels, -100)

        input_ids = torch.tensor(batch_input_ids)
        attention_mask = torch.tensor(batch_attention_mask)
        position_ids = torch.tensor(batch_position_ids)
        labels = torch.tensor(batch_labels)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'labels': labels
        }
