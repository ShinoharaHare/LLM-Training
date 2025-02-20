from typing import Any

import tokenizers
from datasets import Features, Sequence, Value
from packaging.version import Version
from transformers import PreTrainedTokenizerBase

from llm_training.data.hf_based.hf_based_datamodule import (DatasetDict,
                                                            HFBasedDataModule)

from .preference_tuning_datacollator import PreferenceTuningDataCollator
from .preference_tuning_datamodule_config import (
    OverlongHandlingMethod, PreferenceTuningDataModuleConfig)


class PreferenceTuningDataModule(HFBasedDataModule):
    config: PreferenceTuningDataModuleConfig
    datacollator_class = PreferenceTuningDataCollator

    def __init__(self, config: PreferenceTuningDataModuleConfig) -> None:
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
        chat_template: str | None
    ):
        new_batch = {
            'chosen_input_ids': [],
            'chosen_labels': [],
            'chosen_length': [],
            'rejected_input_ids': [],
            'rejected_labels': [],
            'rejected_length': []
        }

        chosen_messages = []
        rejected_messages = []
        for prompt, chosen, rejected in zip(
            batch['prompt'],
            batch['chosen'],
            batch['rejected']
        ):
            chosen_messages.append([
                {'role': 'user', 'content': prompt},
                {'role': 'assistant', 'content': chosen}
            ])

            rejected_messages.append([
                {'role': 'user', 'content': prompt},
                {'role': 'assistant', 'content': rejected}
            ])

        kwargs = dict(
            chat_template=chat_template,
            return_dict=True,
            return_assistant_tokens_mask=True,
            tokenizer_kwargs=dict(
                return_attention_mask=False,
                verbose=False
            )
        )

        chosen_batch_encoding = tokenizer.apply_chat_template(chosen_messages, **kwargs)
        for input_ids, assistant_masks in zip(
            chosen_batch_encoding['input_ids'],
            chosen_batch_encoding['assistant_masks']
        ):
            labels = [i if a == 1 else -100 for i, a in zip(input_ids, assistant_masks)]
            new_batch['chosen_input_ids'].append(input_ids)
            new_batch['chosen_labels'].append(labels)
            new_batch['chosen_length'].append(len(input_ids))

        rejected_batch_encoding = tokenizer.apply_chat_template(rejected_messages, **kwargs)
        for input_ids, assistant_masks in zip(
            rejected_batch_encoding['input_ids'],
            rejected_batch_encoding['assistant_masks']
        ):
            labels = [i if a == 1 else -100 for i, a in zip(input_ids, assistant_masks)]
            new_batch['rejected_input_ids'].append(input_ids)
            new_batch['rejected_labels'].append(labels)
            new_batch['rejected_length'].append(len(input_ids))

        return new_batch
    
    @classmethod
    def _drop_overlong_examples(
        cls,
        batch: dict[str, Any],
        max_length: int
    ):
        indices = [
            i for i in range(len(batch['chosen_length']))
            if max(batch['chosen_length'][i], batch['rejected_length'][i]) <= max_length
        ]
        return {k: [v[i] for i in indices] for k, v in batch.items()}
    
    @classmethod
    def _pre_process_data(
        cls,
        batch: dict[str, Any],
        tokenizer: PreTrainedTokenizerBase,
        chat_template: str | None,
        max_length: int | None,
        overlong_handling_method: OverlongHandlingMethod
    ):
        batch = cls._apply_chat_template_and_tokenize(
            batch,
            tokenizer=tokenizer,
            chat_template=chat_template
        )

        if max_length is not None:
            if overlong_handling_method == OverlongHandlingMethod.DROP:
                batch = cls._drop_overlong_examples(batch, max_length)
        
        return batch
 
    def pre_process_data(self, dataset_dict: DatasetDict) -> DatasetDict:
        dataset_dict = self.map_dataset_dict(
            dataset_dict,
            self._pre_process_data,
            fn_kwargs=dict(
                tokenizer=self.config.tokenizer,
                chat_template=self.config.chat_template,
                max_length=self.config.max_length,
                overlong_handling_method=self.config.overlong_handling_method
            ),
            batched=True,
            remove_columns=True,
            num_proc=self.config.num_proc,
            features=Features({
                'chosen_input_ids': Sequence(Value('int32')),
                'chosen_labels': Sequence(Value('int32')),
                'chosen_length': Value('uint32'),
                'rejected_input_ids': Sequence(Value('int32')),
                'rejected_labels': Sequence(Value('int32')),
                'rejected_length': Value('uint32')
            }),
            desc='Pre-processing data'
        )
        return dataset_dict
