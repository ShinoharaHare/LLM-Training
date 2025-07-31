import logging
from enum import auto

from pydantic import field_validator
from transformers import PreTrainedTokenizerBase

from llm_training.data.chat_templates import get_chat_template
from llm_training.data.hf_based.hf_based_datamodule_config import \
    HFBasedDataModuleConfig
from llm_training.utils.str_enum import StrEnum

logger = logging.getLogger(__name__)


class OverlongHandlingMethod(StrEnum):
    DROP = auto()


class PreferenceTuningDataModuleConfig(HFBasedDataModuleConfig):
    tokenizer: PreTrainedTokenizerBase
    chat_template: str | None = None
    max_length: int | None = None
    overlong_handling_method: OverlongHandlingMethod | str = OverlongHandlingMethod.DROP
    pad_to_multiple_of: int | None = None
    pad_to_max_length: bool = False

    @field_validator('chat_template')
    @classmethod
    def validate_chat_template(cls, value: str | None) -> str | None:
        if value is not None:
            value = get_chat_template(value)
        return value
