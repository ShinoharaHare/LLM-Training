import os
from pathlib import Path
from typing import Any

import fire
import torch
import yaml
from accelerate import init_empty_weights
from lightning import LightningDataModule, LightningModule
from lightning.pytorch.cli import LightningArgumentParser

from llm_training.data import *
from llm_training.lms import BaseLightningModule
from llm_training.models import HFCompatModel


def main(
    checkpoint_path: str | Path,
    output_dir: str | Path | None = None,
    config_path: str | None = None,
    eos_token_id: int | list[int] | None = None,
    dtype: torch.dtype | None = None
) -> None:
    checkpoint_path = Path(checkpoint_path)

    if output_dir is None:
        output_dir = checkpoint_path.parent / 'hf' / checkpoint_path.stem
    else:
        output_dir = Path(output_dir)

    print('Converting checkpoint')
    checkpoint = convert_checkpoint(checkpoint_path)

    if config_path is not None:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        config = checkpoint['config']

    dtype = dtype or get_dtype_from_config(config)
    
    lightning_module, datamodule = instantiate_model_and_datamodule(config)

    assert isinstance(lightning_module, BaseLightningModule)

    lightning_module.config.load_weights = False
    lightning_module.config.init_weights = False
    with init_empty_weights(include_buffers=False):
        lightning_module.configure_model()
    
    state_dict = checkpoint['state_dict']
    required_keys = lightning_module.required_keys
    missing_keys = required_keys - state_dict.keys()
    if len(missing_keys) > 0:
        original_state_dict = lightning_module.get_pre_trained_weights()
        state_dict |= {k: original_state_dict[k] for k in missing_keys}

    incompatiable_keys = lightning_module.load_state_dict(state_dict, strict=False, assign=True)
    missing_keys = set(incompatiable_keys.missing_keys)
    required_missing_keys = missing_keys & required_keys
    assert len(missing_keys & required_keys) == 0, f"Missing keys: {required_missing_keys}"
    
    model = lightning_module.get_model()
    assert isinstance(model, HFCompatModel), f"{model.__class__} is not supported to be converted to HF version."
    model.config.torch_dtype = dtype
    hf_model = model.get_hf_model()
    
    if eos_token_id is not None:
        hf_model.config.eos_token_id = eos_token_id
        hf_model.generation_config.eos_token_id = eos_token_id
    
    print('Saving model')
    hf_model.to(dtype).save_pretrained(output_dir)
    
    if isinstance(
        datamodule,
        (
            PreTrainingDataModule,
            InstructionTuningDataModule,
            PreferenceTuningDataModule,
            VisualInstructionTuningDataModule
        )
    ):
        print('Saving tokenizer')
        tokenizer = datamodule.config.tokenizer
        tokenizer.model_max_length = max(tokenizer.model_max_length, datamodule.config.max_length)
        chat_template = getattr(datamodule.config, 'chat_template', None)
        if chat_template is not None:
            tokenizer.chat_template = chat_template
        tokenizer.save_pretrained(output_dir)
    
    if isinstance(datamodule, VisualInstructionTuningDataModule):
        print('Saving image processor')
        image_processor = datamodule.config.image_processor
        image_processor.save_pretrained(output_dir)


def convert_checkpoint(path: Path) -> dict[str, Any]:
    if (
        path.is_dir()
        and path.joinpath('checkpoint').is_dir()
        and path.joinpath('latest').is_file()
        and path.joinpath('zero_to_fp32.py').is_file()
    ):
        print('DeepSpeed checkpoint detected')
        from lightning.pytorch.utilities.deepspeed import \
            convert_zero_checkpoint_to_fp32_state_dict
        return convert_zero_checkpoint_to_fp32_state_dict(path, os.devnull)
    
    if (
        path.is_dir()
        and path.joinpath('meta.pt').is_file()
        and len(list(path.glob('*.distcp'))) > 0
    ):        
        print('FSDP checkpoint detected')
        return convert_fsdp_checkpoint(path)

    return torch.load(path, 'cpu')


def convert_fsdp_checkpoint(path: Path) -> dict[str, Any]:
    from lightning.fabric.utilities.load import _METADATA_FILENAME
    from torch.distributed.checkpoint import FileSystemReader, load
    from torch.distributed.checkpoint.metadata import (BytesStorageMetadata,
                                                       TensorStorageMetadata)
    
    reader = FileSystemReader(path)
    metadata = reader.read_metadata()

    tensor_names = [n for n in metadata.state_dict_metadata.keys() if n.startswith('model.')]
    state_dict = {}
    for tensor_name in tensor_names:
        sd_metadata = metadata.state_dict_metadata[tensor_name]

        if isinstance(sd_metadata, BytesStorageMetadata):
            state_dict[tensor_name] = '<bytes_io>'
        elif isinstance(sd_metadata, TensorStorageMetadata):
            state_dict[tensor_name] = torch.empty(
                size=sd_metadata.size,
                dtype=sd_metadata.properties.dtype,
                device=torch.device('cpu'),
                memory_format=sd_metadata.properties.memory_format,
                layout=sd_metadata.properties.layout,
                requires_grad=sd_metadata.properties.requires_grad,
                pin_memory=sd_metadata.properties.pin_memory
            )
        else:
            raise NotImplementedError()

    load(state_dict=state_dict, storage_reader=reader)
    
    state_dict = {k.removeprefix('model.'): v for k, v in state_dict.items()}
    checkpoint = {'state_dict': state_dict}
    # This is the extra file saved by Fabric, with user data separate from weights and optimizer states
    extra_file = path / _METADATA_FILENAME
    extra = torch.load(extra_file, map_location='cpu') if extra_file.is_file() else {}
    checkpoint.update(extra)

    return checkpoint


def instantiate_model_and_datamodule(config: dict[str, Any]) -> tuple[LightningModule, LightningDataModule]:
    parser = LightningArgumentParser()
    parser.add_lightning_class_args(LightningModule, 'model', subclass_mode=True)
    parser.add_lightning_class_args(LightningDataModule, 'data', subclass_mode=True)    
    classes = parser.instantiate_classes({k: config[k] for k in ['model', 'data']})
    return classes['model'], classes['data']


def get_dtype_from_config(config: dict[str, Any]) -> torch.dtype:
    dtype_mapping = {
        '16-true': torch.half,
        'bf16-true': torch.bfloat16
    }
    return dtype_mapping.get(config['trainer']['precision'], torch.float)


if __name__ == '__main__':
    fire.Fire(main)
