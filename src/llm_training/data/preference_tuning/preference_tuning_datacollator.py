import math
from typing import Any, TypeVar

import torch
from llm_training.data.base_datacollator import BaseDataCollator

from .preference_tuning_datamodule_config import \
    PreferenceTuningDataModuleConfig

T = TypeVar('T')

class PreferenceTuningDataCollator(BaseDataCollator):
    config: PreferenceTuningDataModuleConfig

    def __init__(self, config: PreferenceTuningDataModuleConfig):
        super().__init__(config)

        assert 'pad_token' in config.tokenizer.special_tokens_map, \
            '`pad_token` is not specified. Please set it manually.'

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

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        outputs = {
            'chosen_input_ids': [],
            'chosen_attention_mask': [],
            'chosen_labels': [],
            'rejected_input_ids': [],
            'rejected_attention_mask': [],
            'rejected_labels': []
        }
        
        for x in batch:
            outputs['chosen_input_ids'].append(x['chosen_input_ids'])
            outputs['chosen_attention_mask'].append([1] * len(x['chosen_input_ids']))
            outputs['chosen_labels'].append(x['chosen_labels'])
            outputs['rejected_input_ids'].append(x['rejected_input_ids'])
            outputs['rejected_attention_mask'].append([1] * len(x['rejected_input_ids']))
            outputs['rejected_labels'].append(x['rejected_labels'])

        outputs['chosen_input_ids'] = self._pad_to_longest(outputs['chosen_input_ids'], self.config.tokenizer.pad_token_id)
        outputs['chosen_attention_mask'] = self._pad_to_longest(outputs['chosen_attention_mask'], 0)
        outputs['chosen_labels'] = self._pad_to_longest(outputs['chosen_labels'], -100)
        outputs['rejected_input_ids'] = self._pad_to_longest(outputs['rejected_input_ids'], self.config.tokenizer.pad_token_id)
        outputs['rejected_attention_mask'] = self._pad_to_longest(outputs['rejected_attention_mask'], 0)
        outputs['rejected_labels'] = self._pad_to_longest(outputs['rejected_labels'], -100)

        outputs['chosen_input_ids'] = torch.tensor(outputs['chosen_input_ids'])
        outputs['chosen_position_ids'] = torch.arange(outputs['chosen_input_ids'].size(1)).unsqueeze(0)
        outputs['chosen_attention_mask'] = torch.tensor(outputs['chosen_attention_mask'])
        outputs['chosen_labels'] = torch.tensor(outputs['chosen_labels'])
        outputs['rejected_input_ids'] = torch.tensor(outputs['rejected_input_ids'])
        outputs['rejected_position_ids'] = torch.arange(outputs['rejected_input_ids'].size(1)).unsqueeze(0)
        outputs['rejected_attention_mask'] = torch.tensor(outputs['rejected_attention_mask'])
        outputs['rejected_labels'] = torch.tensor(outputs['rejected_labels'])

        return outputs
