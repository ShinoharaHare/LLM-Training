# Model Implementations

## Optimized Models

`LLM-Training` has implemented several models that are more efficient compared to those in [Hugging Face](https://github.com/huggingface/transformers) and support additional features.
Therefore, it's recommended to prioritize using these models.

The optimized models currently implemented:

- [Phi-3](/src/llm_training/models/phi3/phi3_model.py)
    - [x] RMS Norm Fusion
    - [x] SwiGLU Fusion
    - [x] RoPE Fusion
    - [x] RoPE Caching
    - [x] Selective Activation Checkpointing
- [LLaMA](/src/llm_training/models/llama/llama_model.py)
    - [x] RMS Norm Fusion
    - [x] SwiGLU Fusion
    - [x] RoPE Fusion
    - [x] RoPE Caching
    - [x] Selective Activation Checkpointing

## Hugging Face Models

If you need to use a model implemented by Hugging Face, you can use [`HFCausalLM`](/src/llm_training/models/hf_causal_lm/hf_causal_lm.py).

```yaml
...
model:
  class_path: llm_training.lms.CLM # or other objective
  init_args.config:
    model:
      model_class: llm_training.models.HFCausalLM
      model_config:
        hf_path: <PRETRAINED_MODEL_NAME_OR_PATH>
        torch_dtype: ???
        attn_implementation: ???
        enable_gradient_checkpointing: ???
...
```
