from jsonargparse import class_from_function
from transformers import (AutoImageProcessor, AutoTokenizer,
                          BaseImageProcessor, PreTrainedTokenizerBase)


def _load_tokenizer(path: str, pad_token: str | None = None, **kwargs) -> PreTrainedTokenizerBase:
    if pad_token is not None:
        kwargs['pad_token'] = pad_token
    return AutoTokenizer.from_pretrained(path, **kwargs)


def _load_image_processor(path: str, pad_token: str | None = None, **kwargs) -> BaseImageProcessor:
    return AutoImageProcessor.from_pretrained(path, **kwargs)


HFTokenizer = class_from_function(_load_tokenizer, name='HFTokenizer')
HFImageProcessor = class_from_function(_load_image_processor, name='HFImageProcessor')
