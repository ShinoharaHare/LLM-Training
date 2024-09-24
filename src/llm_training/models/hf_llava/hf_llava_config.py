from llm_training.models.hf_compat_model import HFCompatModelConfig

class HFLlavaConfig(HFCompatModelConfig):
    enable_gradient_checkpointing: bool = False
