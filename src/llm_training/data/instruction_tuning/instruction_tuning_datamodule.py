import random
from typing import Any

import tokenizers
from datasets import Features, Sequence, Value
from packaging.version import Version
from transformers import PreTrainedTokenizerBase

from llm_training.data.hf_based.hf_based_datamodule import (DatasetDict,
                                                            HFBasedDataModule)

from .instruction_tuning_datacollator import InstructionTuningDataCollator
from .instruction_tuning_datamodule_config import (
    InstructionTuningDataModuleConfig, OverlongHandlingMethod, PackingMethod)


class InstructionTuningDataModule(HFBasedDataModule):
    config: InstructionTuningDataModuleConfig
    datacollator_class = InstructionTuningDataCollator

    def __init__(self, config: InstructionTuningDataModuleConfig) -> None:
        super().__init__(config)

        if Version(tokenizers.__version__) < Version('0.20.1'):
            raise ValueError(
                "`tokenizers` must be at least version 0.20.1, "
                "otherwise LLaMA 3 tokenizer will produce incorrect prompt/response mask."
            )

    @classmethod
    def _apply_chat_template_and_tokenize(
        cls,
        batch: dict[str, list[str]],
        tokenizer: PreTrainedTokenizerBase,
        chat_template: str | None,
        default_system_prompt: str | None,
        add_default_system_prompt_rate: float | None
    ):
        new_batch = {
            'input_ids': [],
            'attention_mask': [],
            'labels': [],
            'length': []
        }

        for messages in batch['messages']:
            # Add an empty system prompt randomly if it does not exist.
            has_system_prompt = any(m['role'] == 'system' for m in messages)
            if (
                not has_system_prompt
                and default_system_prompt is not None
                and add_default_system_prompt_rate is not None
                and random.random() < add_default_system_prompt_rate
            ):
                messages.insert(0, {'role': 'system', 'content': default_system_prompt})

        batch_encoding = tokenizer.apply_chat_template(
            batch['messages'],
            chat_template=chat_template,
            return_dict=True,
            return_assistant_tokens_mask=True,
            tokenizer_kwargs=dict(
                return_attention_mask=False,
                verbose=False
            )
        )
        
        for input_ids, assistant_masks in zip(
            batch_encoding['input_ids'],
            batch_encoding['assistant_masks']
        ):
            labels = [i if a == 1 else -100 for i, a in zip(input_ids, assistant_masks)]
            new_batch['input_ids'].append(input_ids)
            new_batch['attention_mask'].append([1] * len(input_ids))
            new_batch['labels'].append(labels)
            new_batch['length'].append(len(input_ids))

        return new_batch
    
    @classmethod
    def _drop_overlong_examples(
        cls,
        batch: dict[str, Any],
        max_length: int
    ):
        indices = [i for i, n in enumerate(batch['length']) if n <= max_length]
        return {k: [v[i] for i in indices] for k, v in batch.items()}
    
    @classmethod
    def _truncate_overlong_examples(
        cls,
        batch: dict[str, Any],
        max_length: int
    ):
        for i in range(len(batch['input_ids'])):
            if batch['length'][i] > max_length:
                batch['input_ids'][i] = batch['input_ids'][:max_length]
                batch['labels'][i] = batch['labels'][:max_length]
                batch['length'][i] = max_length
        return batch
    
    @classmethod
    def _group_indices_by_length(cls, lengths: list[int], max_length: int) -> list[list[int]]:
        groups = []
        current_group = []
        current_sum = 0
        
        for i, l in sorted(enumerate(lengths), key=lambda x: x[1]):
            if current_sum + l + len(current_group) <= max_length:
                current_group.append(i)
                current_sum += l
            else:
                groups.append(current_group)
                current_group = [i]
                current_sum = l
        
        if current_group:
            groups.append(current_group)
        
        return groups

    @classmethod
    def _group_by_length(cls, batch: dict[str, list[list[int]]], max_length: int):
        new_batch = {
            'input_ids': [],
            'attention_mask': [],
            'labels': [],
            'length': []
        }

        groups = cls._group_indices_by_length(batch['length'], max_length)
        for group in groups:
            input_ids = []
            attention_mask = []
            labels = []
            for local_idx, global_idx in enumerate(group):
                input_ids += batch['input_ids'][global_idx]
                attention_mask += [local_idx + 1] * batch['length'][global_idx]
                labels += batch['labels'][global_idx]
            new_batch['input_ids'].append(input_ids)
            new_batch['attention_mask'].append(attention_mask)
            new_batch['labels'].append(labels)
            new_batch['length'].append(len(input_ids))

        return new_batch
    
    @classmethod
    def _pre_process_data(
        cls,
        batch: dict[str, list],
        tokenizer: PreTrainedTokenizerBase,
        chat_template: str | None,
        default_system_prompt: str | None,
        add_default_system_prompt_rate: float | None,
        max_length: int | None,
        overlong_handling_method: OverlongHandlingMethod,
        packing_method: PackingMethod
    ) -> dict[str, list]:
        batch = cls._apply_chat_template_and_tokenize(
            batch,
            tokenizer=tokenizer,
            chat_template=chat_template,
            default_system_prompt=default_system_prompt,
            add_default_system_prompt_rate=add_default_system_prompt_rate
        )

        if max_length is not None:
            if overlong_handling_method == OverlongHandlingMethod.DROP:
                batch = cls._drop_overlong_examples(batch, max_length)
            elif overlong_handling_method == OverlongHandlingMethod.TRUNCATE:
                batch = cls._truncate_overlong_examples(batch, max_length)

            if packing_method == PackingMethod.GROUP_BY_LENGTH:
                batch = cls._group_by_length(batch, max_length)

        return batch
 
    def pre_process_data(self, dataset_dict: DatasetDict) -> DatasetDict:
        dataset_dict = self.map_dataset_dict(
            dataset_dict,
            self._pre_process_data,
            fn_kwargs=dict(
                tokenizer=self.config.tokenizer,
                chat_template=self.config.chat_template,
                default_system_prompt=self.config.default_system_prompt,
                add_default_system_prompt_rate=self.config.add_default_system_prompt_rate,
                max_length=self.config.max_length,
                overlong_handling_method=self.config.overlong_handling_method,
                packing_method=self.config.packing_method
            ),
            batched=True,
            remove_columns=True,
            num_proc=self.config.num_proc,
            features=Features({
                'input_ids': Sequence(Value('int32')),
                'attention_mask': Sequence(Value('uint16')),
                'labels': Sequence(Value('int32')),
                'length': Value('uint32')
            }),
            desc='Pre-processing data'
        )
        return dataset_dict
