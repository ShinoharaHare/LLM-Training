from typing import Any

from transformers import BaseImageProcessor, PreTrainedTokenizerBase

from llm_training.data.hf_based.hf_based_datamodule import (DatasetDict,
                                                            HFBasedDataModule)

from .visual_instruction_tuning_datacollator import \
    VisualInstructionTuningDataCollator
from .visual_instruction_tuning_datamodule_config import \
    VisualInstructionTuningDataModuleConfig


class VisualInstructionTuningDataModule(HFBasedDataModule):
    config: VisualInstructionTuningDataModuleConfig
    datacollator_class = VisualInstructionTuningDataCollator

    def __init__(self, config: VisualInstructionTuningDataModuleConfig) -> None:
        super().__init__(config)
 
    def pre_process_data(self, dataset_dict: DatasetDict) -> DatasetDict:
        dataset_dict = self.map_dataset_dict(
            dataset_dict,
            _process_text_and_image,
            remove_columns=True,
            fn_kwargs=dict(
                tokenizer=self.config.tokenizer,
                chat_template=self.config.chat_template,
                image_processor=self.config.image_processor
            ),
            num_proc=self.config.num_proc,
            desc='Process text and image'
        )

        dataset_dict = dataset_dict.filter(
            _drop_overlong,
            input_columns='input_ids',
            fn_kwargs=dict(max_length=self.config.max_length),
            num_proc=self.config.num_proc,
            desc='Drop overlong'
        )
    
        return dataset_dict


def _process_text_and_image(
    example: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    chat_template: str | None,
    image_processor: BaseImageProcessor
):
    messages = example['messages']
    image = example['image']

    input_ids = []
    labels = []

    system_prompt = None
    if messages[0]['role'] == 'system':
        system_prompt = messages.pop(0)

    for i, message in enumerate(messages):
        conversation = [message]
        if i == 0 and system_prompt is not None:
            conversation.insert(0, system_prompt)
        text = tokenizer.apply_chat_template(
            conversation,
            chat_template=chat_template,
            tokenize=False,
            add_generation_prompt=message['role'] == 'user',
            index=i,
            length=len(messages)
        )
        # 這裡將同一筆資料分多次 tokenize，為保證跟一次 tokenize 全部的結果相同
        # 先在前面加一個 token，encode 後再移除掉
        text = tokenizer.bos_token + text
        current_input_ids = tokenizer.encode(text, add_special_tokens=False)
        current_input_ids = current_input_ids[1:]
        
        if message['role'] in ['system', 'user']:
            input_ids += current_input_ids
            labels += [-100] * len(current_input_ids)
        elif message['role'] == 'assistant':
            input_ids += current_input_ids
            labels += current_input_ids
        else:
            raise ValueError(f"Unknown role: `{message['role']}`")

    pixel_values = image_processor(image).pixel_values[0]

    return {
        'input_ids': input_ids,
        'labels': labels,
        'pixel_values': pixel_values
    }


def _drop_overlong(input_ids: list[int], max_length: int):
    return len(input_ids) <= max_length
