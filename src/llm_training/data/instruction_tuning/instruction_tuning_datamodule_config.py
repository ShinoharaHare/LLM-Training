import logging
from enum import auto

from pydantic import ValidationInfo, field_validator
from transformers import PreTrainedTokenizerBase

from llm_training.data.chat_templates import get_chat_template
from llm_training.data.hf_based.hf_based_datamodule_config import \
    HFBasedDataModuleConfig
from llm_training.utils.str_enum import StrEnum

logger = logging.getLogger(__name__)


class OverlongHandlingMethod(StrEnum):
    DROP = auto()
    TRUNCATE = auto()


class PackingMethod(StrEnum):
    NO_PACKING = auto()
    GROUP_BY_LENGTH = auto()


class InstructionTuningDataModuleConfig(HFBasedDataModuleConfig):
    tokenizer: PreTrainedTokenizerBase
    chat_template: str | None = None
    max_length: int | None = None
    overlong_handling_method: OverlongHandlingMethod | str = OverlongHandlingMethod.DROP
    packing_method: PackingMethod | str = PackingMethod.NO_PACKING
    pad_to_multiple_of: int | None = None
    pad_to_max_length: bool = False
    add_default_system_prompt_rate: float | None = None
    default_system_prompt: str | None = None

    @field_validator('chat_template')
    @classmethod
    def validate_chat_template(cls, value: str | None) -> str | None:
        if value is not None:
            value = get_chat_template(value)
        return value

    @field_validator('default_system_prompt')
    @classmethod
    def validate_default_system_prompt(cls, value: str | None, info: ValidationInfo) -> str | None:
        assert value is None or info.data['add_default_system_prompt_rate'] is not None, \
            "Default system prompt must be set to use `add_default_system_prompt_rate`."
        return value

    @field_validator('overlong_handling_method')
    @classmethod
    def validate_overlong_handling_method(cls, value: OverlongHandlingMethod | str) -> OverlongHandlingMethod:
        return OverlongHandlingMethod(value.lower())

    @field_validator('packing_method')
    @classmethod
    def validate_packing_method(cls, value: PackingMethod | str) -> PackingMethod:
        return PackingMethod(value.lower())
