from enum import auto

from pydantic import Field, ValidationInfo, field_validator
from transformers import PreTrainedTokenizerBase

from llm_training.data.hf_based import HFBasedDataModuleConfig
from llm_training.utils.str_enum import StrEnum


class PackingMethod(StrEnum):
    NO_PACKING = auto()
    NAIVE_PACKING = auto()
    BEST_FIT_BIN_PACKING = auto()


class PreTrainingDataModuleConfig(HFBasedDataModuleConfig):
    tokenizer: PreTrainedTokenizerBase
    max_length: int | None = None
    stride: int | None = None
    packing_method: PackingMethod | str = PackingMethod.NAIVE_PACKING
    sample_rate: dict[str, float] = Field(default_factory=dict)
    pre_processing_batch_size: int = 1000
    pad_to_multiple_of: int | None = None
    pad_to_max_length: bool = False

    @field_validator('packing_method')
    @classmethod
    def validate_packing_method(cls, value: PackingMethod | str, info: ValidationInfo) -> PackingMethod:
        value = PackingMethod(value.lower())
        assert value in (PackingMethod.NAIVE_PACKING, PackingMethod.BEST_FIT_BIN_PACKING) or info.data['max_length'] is not None, \
            "You must set `max_length` to packing data"
        return value
    
    @field_validator('stride')
    @classmethod
    def validate_stride(cls, value: int | None, info: ValidationInfo) -> int | None:
        max_length = info.data['max_length']

        if value is None:
            value = max_length
        else:
            assert max_length is not None, "You must also set `max_length` to use `stride`"
            assert value <= max_length, "`stride` must be <= `max_length`"

        return value
