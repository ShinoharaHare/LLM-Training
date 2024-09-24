import logging

from pydantic import field_validator
from transformers import BaseImageProcessor, PreTrainedTokenizerBase

from llm_training.data.chat_templates import get_chat_template
from llm_training.data.hf_based.hf_based_datamodule_config import \
    HFBasedDataModuleConfig

logger = logging.getLogger(__name__)


class VisualInstructionTuningDataModuleConfig(HFBasedDataModuleConfig):
    tokenizer: PreTrainedTokenizerBase
    image_processor: BaseImageProcessor
    chat_template: str | None = None
    max_length: int | None = None
    pad_to_multiple_of: int | None = None

    @field_validator('chat_template')
    @classmethod
    def validate_chat_template(cls, value: str | None) -> str | None:
        if value is not None:
            value = get_chat_template(value)
        return value
