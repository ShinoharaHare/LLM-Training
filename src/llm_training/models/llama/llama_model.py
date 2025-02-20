import logging
import math
from functools import partial

import torch
import torch.utils.checkpoint
from torch import nn
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.parallel import (ColwiseParallel,
                                               PrepareModuleInput,
                                               RowwiseParallel,
                                               SequenceParallel,
                                               parallelize_module)
from transformers import LlamaConfig as HFLlamaConfig
from transformers import LlamaForCausalLM

from llm_training.models.hf_compat_model import HFCompatModel
from llm_training.models.utils.modeling_outputs import CausalLMOutput
from llm_training.ops import *
from llm_training.ops import liger_kernel
from llm_training.ops.attention_op import (flash_attention_forward,
                                           prepare_4d_causal_attention_mask)
from llm_training.ops.rope_utils import ROPE_INIT_FUNCTIONS, RoPEConfig
from llm_training.utils.decorators import copy_method_signature

from .llama_config import LlamaConfig

logger = logging.getLogger(__name__)


class Llama(HFCompatModel):
    config: LlamaConfig
    layers: list["LlamaDecoderLayer"]

    config_class = LlamaConfig
    hf_config_class = HFLlamaConfig
    hf_model_class = LlamaForCausalLM

    no_split_modules = ['LlamaDecoderLayer']

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__(config)

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config)
        self.rotary_emb = LlamaRotaryEmbedding(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def _init_weights_impl(self, module: nn.Module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def merge_hf_config(self, hf_config: HFLlamaConfig) -> None:
        assert hf_config.hidden_act == 'silu'

        self.config.tie_word_embeddings = hf_config.tie_word_embeddings
        self.config.vocab_size = hf_config.vocab_size
        self.config.hidden_size = hf_config.hidden_size
        self.config.intermediate_size = hf_config.intermediate_size
        self.config.num_hidden_layers = hf_config.num_hidden_layers
        self.config.num_attention_heads = hf_config.num_attention_heads
        self.config.num_key_value_heads = hf_config.num_key_value_heads
        self.config.max_position_embeddings = hf_config.max_position_embeddings
        self.config.initializer_range = hf_config.initializer_range
        self.config.rms_norm_eps = hf_config.rms_norm_eps
        self.config.pad_token_id = hf_config.pad_token_id
        self.config.bos_token_id = hf_config.bos_token_id
        self.config.eos_token_id = hf_config.eos_token_id
        self.config.rope_theta = hf_config.rope_theta
        self.config.attention_dropout = hf_config.attention_dropout
        self.config.attention_bias = hf_config.attention_bias
        self.config.mlp_bias = hf_config.mlp_bias

    def convert_state_dict_from_hf(self, hf_state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {k.removeprefix('model.'): v for k, v in hf_state_dict.items()}
    
    def convert_state_dict_to_hf(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {('' if 'lm_head' in k else 'model.') + k: v for k, v in state_dict.items()}
    
    def forward_decoder_layer(
        self,
        layer: "LlamaDecoderLayer",
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        if (
            self.config.enable_gradient_checkpointing
            and self.config.recompute_granularity == 'full'
            and self.training
        ):
            return torch.utils.checkpoint.checkpoint(
                layer.__call__,
                hidden_states,
                position_embeddings,
                attention_mask,
                use_reentrant=False
            )
        return layer(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask
        )

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens
    
    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head
    
    def set_input_embeddings(self, embedding: nn.Embedding) -> None:
        self.embed_tokens = embedding

    def set_output_embeddings(self, linear: nn.Linear) -> None:
        self.lm_head = linear

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        return_last_hidden_states: bool = False
    ) -> CausalLMOutput:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self.config._attn_implementation == 'flash_attention_2':
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        else:
            # 4d mask is passed through the layers
            attention_mask = prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                0
            )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for decoder_layer in self.layers:
            hidden_states = self.forward_decoder_layer(
                decoder_layer,
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask
            )

        last_hidden_states = hidden_states if return_last_hidden_states else None
        hidden_states = self.norm(hidden_states)

        logits = self.lm_head(hidden_states)

        return CausalLMOutput(
            logits=logits,
            last_hidden_states=last_hidden_states
        )

    @copy_method_signature(forward)
    def __call__(): ...

    def configure_tensor_parallel(self, tp_mesh):
        if tp_mesh.size() == 1:
            return

        parallelize_module(
            self,
            tp_mesh,
            {
                'embed_tokens': RowwiseParallel(
                    input_layouts=Replicate(),
                    output_layouts=Shard(1)
                ),
                'norm': SequenceParallel(),
                'lm_head': ColwiseParallel(
                    input_layouts=Shard(1),
                    use_local_output=False
                )
            }
        )

        for layer in self.layers:
            parallelize_module(
                layer,
                tp_mesh,
                {
                    'input_layernorm': SequenceParallel(),
                    'self_attn': PrepareModuleInput(
                        input_kwarg_layouts={'hidden_states': Shard(1)},
                        desired_input_kwarg_layouts={'hidden_states': Replicate()}
                    ),
                    'self_attn.q_proj': ColwiseParallel(),
                    'self_attn.k_proj': ColwiseParallel(),
                    'self_attn.v_proj': ColwiseParallel(),
                    'self_attn.o_proj': RowwiseParallel(output_layouts=Shard(1)),
                    'post_attention_layernorm': SequenceParallel(),
                    'mlp': PrepareModuleInput(
                        input_layouts=(Shard(1),),
                        desired_input_layouts=(Replicate(),)
                    ),
                    'mlp.gate_proj': ColwiseParallel(),
                    'mlp.up_proj': ColwiseParallel(),
                    'mlp.down_proj': RowwiseParallel(output_layouts=Shard(1))
                }
            )

            self_attn = layer.self_attn
            self_attn.num_heads //= tp_mesh.size()
            self_attn.num_key_value_heads //= tp_mesh.size()

    def configure_fully_sharded_data_parallel(self, dp_mesh, reshard_after_forward, mp_policy, offload_policy, **kwargs):
        if dp_mesh.size() == 1:
            return
        
        fully_shard_ = partial(
            fully_shard,
            mesh=dp_mesh,
            reshard_after_forward=reshard_after_forward,
            mp_policy=mp_policy,
            offload_policy=offload_policy
        )

        # shard the entire model to support gradient clipping.
        # see the following issues:
        # https://github.com/pytorch/pytorch/issues/121020
        # https://github.com/pytorch/pytorch/issues/134212
        
        fully_shard_(self.embed_tokens)
        fully_shard_(self.norm)
        fully_shard_(self.lm_head)
        
        for layer in self.layers:
            fully_shard_(layer)


class LlamaRMSNorm(nn.Module):
    def __init__(self, config: LlamaConfig):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.hidden_size))
        self.variance_epsilon = config.rms_norm_eps

    def forward(self, hidden_states: torch.Tensor):
        if isinstance(hidden_states, DTensor):
            return rms_norm(hidden_states, self.weight, eps=self.variance_epsilon)
        return liger_kernel.rms_norm(hidden_states, self.weight, eps=self.variance_epsilon)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class LlamaRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor
    
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()

        self.config = config
        
        if config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get('rope_type', config.rope_scaling.get('type'))
        else:
            self.rope_type = 'default'
        
        self.rope_config = RoPEConfig(
            type=self.rope_type,
            base=config.rope_theta,
            dim=config.hidden_size // config.num_attention_heads,
            max_position_embeddings=config.max_position_embeddings,
            scaling_config=config.rope_scaling
        )

        self.max_seq_len_cached = 0
        self.original_max_seq_len = config.max_position_embeddings

        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        self.attention_scaling = None
        self.register_buffer('inv_freq', None, persistent=False)

        self._set_cos_sin_cache(
            self.original_max_seq_len,
            device=torch.device('cpu'),
            dtype=torch.float
        )

        self.original_inv_freq = self.inv_freq

    def _set_inv_freq_cache(self, seq_len: int, device: torch.device) -> None:
        if self.rope_type == 'dynamic':
            if seq_len > self.max_seq_len_cached:  # growth
                inv_freq, self.attention_scaling = self.rope_init_fn(
                    self.rope_config,
                    device=device,
                    seq_len=seq_len
                )
                self.register_buffer('inv_freq', inv_freq, persistent=False)  # TODO joao: may break with compilation
                self.max_seq_len_cached = seq_len
            
            if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
                self.register_buffer('inv_freq', self.original_inv_freq, persistent=False)
                self.max_seq_len_cached = self.original_max_seq_len
        elif self.rope_type == 'longrope':
            if seq_len > self.max_seq_len_cached:
                inv_freq, self.attention_scaling = self.rope_init_fn(
                    self.rope_config,
                    device=device,
                    seq_len=seq_len
                )
                self.register_buffer('inv_freq', inv_freq, persistent=False)
            elif seq_len <= self.original_max_seq_len:
                self.register_buffer('inv_freq', self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = seq_len
        else:
            if self.inv_freq is None or self.attention_scaling is None:
                inv_freq, self.attention_scaling = self.rope_init_fn(
                    self.rope_config,
                    device=device,
                    seq_len=seq_len
                )
                self.register_buffer('inv_freq', inv_freq, persistent=False)

            self.max_seq_len_cached = seq_len

        if self.inv_freq.device != device:
            self.inv_freq.data = self.inv_freq.data.to(device)
    
    def _set_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        if seq_len <= self.max_seq_len_cached and seq_len >= self.original_max_seq_len:
            return

        seq_len = math.ceil(seq_len / 4096) * 4096

        self._set_inv_freq_cache(seq_len, device)
        
        device_type = device.type if isinstance(device.type, str) and device.type != 'mps' else 'cpu'
        with torch.autocast(device_type=device_type, enabled=False):
            t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).float()
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)

        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        cos = cos.to(dtype=dtype, device=device)
        sin = sin.to(dtype=dtype, device=device)

        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cos.device != x.device:
            self.cos = self.cos.to(x.device)

        if self.sin.device != x.device:
            self.sin = self.sin.to(x.device)

        seq_len = position_ids.max() + 1

        self._set_cos_sin_cache(
            seq_len.item(),
            device=x.device,
            dtype=x.dtype
        )
        
        return (
            self.cos[position_ids].to(x),
            self.sin[position_ids].to(x)
        )


class LlamaMLP(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        # self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor):
        return self.down_proj(liger_kernel.silu_mul(self.gate_proj(x), self.up_proj(x)))


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    return hidden_states.repeat_interleave(n_rep, dim=1)


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: int | None = None):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)

    def _core_attention_forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        bsz = query_states.size(0)
        q_len = query_states.size(2)
        kv_seq_len = key_states.size(2)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(value_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        return attn_output

    def core_attention_forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        if (
            self.config.enable_gradient_checkpointing
            and self.config.recompute_granularity == 'selective'
            and self.training
        ):
            attn_output = torch.utils.checkpoint.checkpoint(
                self._core_attention_forward,
                query_states,
                key_states,
                value_states,
                attention_mask,
                use_reentrant=False
            )
        else:
            attn_output = self._core_attention_forward(
                query_states,
                key_states,
                value_states,
                attention_mask
            )

        return attn_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rope(query_states, key_states, cos, sin)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_output = self.core_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask
        )

        attn_output = self.o_proj(attn_output)

        return attn_output


class LlamaFlashAttention2(LlamaAttention):
    """
    Llama flash attention module. This module inherits from `LlamaAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def _core_attention_forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        bsz = query_states.size(0)
        q_len = query_states.size(2)

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            sliding_window=getattr(self, "sliding_window", None),
            use_top_left_mask=False,
            is_causal=self.is_causal,
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()

        torch.cuda.synchronize()

        return attn_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rope(query_states, key_states, cos, sin)

        attn_output = self.core_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask
        )
        attn_output = self.o_proj(attn_output)

        return attn_output


class LlamaSdpaAttention(LlamaAttention):
    """
    Llama attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `LlamaAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    def _core_attention_forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        bsz = query_states.size(0)
        q_len = query_states.size(2)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        is_causal = True if causal_mask is None and q_len > 1 else False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)

        return attn_output

    # Adapted from LlamaAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rope(query_states, key_states, cos, sin)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_output = self.core_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask
        )

        attn_output = self.o_proj(attn_output)

        return attn_output


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()

        self.config = config
        self.hidden_size = config.hidden_size

        attention_class = LlamaAttention
        if self.config._attn_implementation == 'flash_attention_2':
            attention_class = LlamaFlashAttention2
        elif self.config._attn_implementation == 'sdpa':
            attention_class = LlamaSdpaAttention
        self.self_attn = attention_class(config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config)
        self.post_attention_layernorm = LlamaRMSNorm(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
