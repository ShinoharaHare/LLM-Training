import inspect
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from flash_attn import flash_attn_func, flash_attn_varlen_func
from flash_attn.bert_padding import (index_first_axis, pad_input,  # noqa
                                     unpad_input)

_flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)

@dataclass
class AttentionMaskConverter:
    """
    A utility attention mask class that allows one to:
        - Create a causal 4d mask
        - Create a causal 4d mask with slided window
        - Convert a 2d attention mask (batch_size, query_length) to a 4d attention mask (batch_size, 1, query_length,
          key_value_length) that can be multiplied with attention scores

    Examples:

    ```python
    >>> import torch
    >>> from transformers.modeling_attn_mask_utils import AttentionMaskConverter

    >>> converter = AttentionMaskConverter(True)
    >>> converter.to_4d(torch.tensor([[0, 0, 0, 1, 1]]), 5, key_value_length=5, dtype=torch.float32)
    tensor([[[[-3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38],
            [-3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38],
            [-3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38],
            [-3.4028e+38, -3.4028e+38, -3.4028e+38,  0.0000e+00, -3.4028e+38],
            [-3.4028e+38, -3.4028e+38, -3.4028e+38,  0.0000e+00,  0.0000e+00]]]])
    ```

    Parameters:
        is_causal (`bool`):
            Whether the attention mask should be a uni-directional (causal) or bi-directional mask.

        sliding_window (`int`, *optional*):
            Optionally, the sliding window masks can be created if `sliding_window` is defined to a positive integer.
    """

    is_causal: bool
    sliding_window: int

    def __init__(self, is_causal: bool, sliding_window: Optional[int] = None):
        self.is_causal = is_causal
        self.sliding_window = sliding_window

        if self.sliding_window is not None and self.sliding_window <= 0:
            raise ValueError(
                f"Make sure that when passing `sliding_window` that its value is a strictly positive integer, not `{self.sliding_window}`"
            )

    def to_causal_4d(
        self,
        batch_size: int,
        query_length: int,
        key_value_length: int,
        dtype: torch.dtype,
        device: Union[torch.device, "str"] = "cpu",
    ) -> Optional[torch.Tensor]:
        """
        Creates a causal 4D mask of (bsz, head_dim=1, query_length, key_value_length) shape and adds large negative
        bias to upper right hand triangular matrix (causal mask).
        """
        if not self.is_causal:
            raise ValueError(f"Please use `to_causal_4d` only if {self.__class__} has `is_causal` set to True.")

        # If shape is not cached, create a new causal mask and cache it
        input_shape = (batch_size, query_length)
        past_key_values_length = key_value_length - query_length

        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        causal_4d_mask = None
        if input_shape[-1] > 1 or self.sliding_window is not None:
            causal_4d_mask = self._make_causal_mask(
                input_shape,
                dtype,
                device=device,
                past_key_values_length=past_key_values_length,
                sliding_window=self.sliding_window,
            )

        return causal_4d_mask

    def to_4d(
        self,
        attention_mask_2d: torch.Tensor,
        query_length: int,
        dtype: torch.dtype,
        key_value_length: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Converts 2D attention mask to 4D attention mask by expanding mask to (bsz, head_dim=1, query_length,
        key_value_length) shape and by adding a large negative bias to not-attended positions. If attention_mask is
        causal, a causal mask will be added.
        """
        input_shape = (attention_mask_2d.shape[0], query_length)

        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        causal_4d_mask = None
        if (input_shape[-1] > 1 or self.sliding_window is not None) and self.is_causal:
            if key_value_length is None:
                raise ValueError(
                    "This attention mask converter is causal. Make sure to pass `key_value_length` to correctly create a causal mask."
                )

            past_key_values_length = key_value_length - query_length
            causal_4d_mask = self._make_causal_mask(
                input_shape,
                dtype,
                device=attention_mask_2d.device,
                past_key_values_length=past_key_values_length,
                sliding_window=self.sliding_window,
            )
        elif self.sliding_window is not None:
            raise NotImplementedError("Sliding window is currently only implemented for causal masking")

        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attn_mask = self._expand_mask(attention_mask_2d, dtype, tgt_len=input_shape[-1]).to(
            attention_mask_2d.device
        )

        if causal_4d_mask is not None:
            expanded_attn_mask = causal_4d_mask.masked_fill(expanded_attn_mask.bool(), torch.finfo(dtype).min)

        # expanded_attn_mask + causal_4d_mask can cause some overflow
        expanded_4d_mask = expanded_attn_mask

        return expanded_4d_mask

    @staticmethod
    def _make_causal_mask(
        input_ids_shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
        past_key_values_length: int = 0,
        sliding_window: Optional[int] = None,
    ):
        """
        Make causal mask used for bi-directional self-attention.
        """
        bsz, tgt_len = input_ids_shape
        mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
        mask_cond = torch.arange(mask.size(-1), device=device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)

        mask = mask.to(dtype)

        if past_key_values_length > 0:
            mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)

        # add lower triangular sliding window mask if necessary
        if sliding_window is not None:
            diagonal = past_key_values_length - sliding_window - 1

            context_mask = torch.tril(torch.ones_like(mask, dtype=torch.bool), diagonal=diagonal)
            mask.masked_fill_(context_mask, torch.finfo(dtype).min)

        return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

    @staticmethod
    def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
        """
        Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
        """
        bsz, src_len = mask.size()
        tgt_len = tgt_len if tgt_len is not None else src_len

        expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

        inverted_mask = 1.0 - expanded_mask

        return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

    @staticmethod
    def _unmask_unattended(
        expanded_mask: torch.FloatTensor,
        min_dtype: float,
    ):
        # fmt: off
        """
        Attend to all tokens in masked rows from the expanded attention mask, for example the relevant first rows when
        using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
        Details: https://github.com/pytorch/pytorch/issues/110213

        `expanded_mask` is [bsz, num_masks, tgt_seq_len, src_seq_len] or [bsz, tgt_seq_len, src_seq_len].
        `attention_mask` is [bsz, src_seq_len].

        The dimension num_masks of `expanded_mask` is most often 1, but it can also be the number of heads in the case of alibi attention bias.

        For example, if `expanded_mask` is (e.g. here left-padding case)
        ```
        [[[[0, 0, 0],
           [0, 0, 0],
           [0, 0, 1]]],
         [[[1, 0, 0],
           [1, 1, 0],
           [1, 1, 1]]],
         [[[0, 0, 0],
           [0, 1, 0],
           [0, 1, 1]]]]
        ```
        then the modified `expanded_mask` will be
        ```
        [[[[1, 1, 1],   <-- modified
           [1, 1, 1],   <-- modified
           [0, 0, 1]]],
         [[[1, 0, 0],
           [1, 1, 0],
           [1, 1, 1]]],
         [[[1, 1, 1],   <-- modified
           [0, 1, 0],
           [0, 1, 1]]]]
        ```
        """
        # fmt: on
        if expanded_mask.dtype == torch.bool:
            raise ValueError(
                "AttentionMaskConverter._unmask_unattended expects a float `expanded_mask`, got a BoolTensor."
            )

        return expanded_mask.mul(~torch.all(expanded_mask == min_dtype, dim=-1, keepdim=True))

    @staticmethod
    def _ignore_causal_mask_sdpa(
        attention_mask: Optional[torch.Tensor],
        inputs_embeds: torch.Tensor,
        past_key_values_length: int,
        sliding_window: Optional[int] = None,
        is_training: bool = False,
    ) -> bool:
        """
        Detects whether the optional user-specified attention_mask & the automatically created causal mask can be ignored in case PyTorch's SDPA is used, rather relying on SDPA's `is_causal` argument.

        In case no token is masked in the `attention_mask` argument, if `query_length == 1` or
        `key_value_length == query_length`, we rather rely on SDPA `is_causal` argument to use causal/non-causal masks,
        allowing to dispatch to the flash attention kernel (that can otherwise not be used if a custom `attn_mask` is passed).
        """

        _, query_length = inputs_embeds.shape[0], inputs_embeds.shape[1]
        key_value_length = query_length + past_key_values_length

        is_tracing = (
            torch.jit.is_tracing()
            or isinstance(inputs_embeds, torch.fx.Proxy)
            or (hasattr(torch, "_dynamo") and torch._dynamo.is_compiling())
        )

        ignore_causal_mask = False

        if attention_mask is None:
            # TODO: When tracing with TorchDynamo with fullgraph=True, the model is recompiled depending on the input shape, thus SDPA's `is_causal` argument is rightfully updated (see https://gist.github.com/fxmarty/1313f39037fc1c112508989628c57363). However, when using `torch.export` or
            # or `torch.onnx.dynamo_export`, we must pass an example input, and `is_causal` behavior is hard-coded. If a user exports a model with q_len > 1, the exported model will hard-code `is_causal=True` which is in general wrong (see https://github.com/pytorch/pytorch/issues/108108).
            # Thus, we only set `ignore_causal_mask = True` if the model is set to training.
            #
            # Besides, jit.trace can not handle the `q_len > 1` condition for `is_causal` ("TypeError: scaled_dot_product_attention(): argument 'is_causal' must be bool, not Tensor").
            if (
                (is_training or not is_tracing)
                and (query_length == 1 or key_value_length == query_length)
                and (sliding_window is None or key_value_length < sliding_window)
            ):
                ignore_causal_mask = True
        elif sliding_window is None or key_value_length < sliding_window:
            if len(attention_mask.shape) == 4:
                return False
            elif (is_training or not is_tracing) and torch.all(attention_mask == 1):
                if query_length == 1 or key_value_length == query_length:
                    # For query_length == 1, causal attention and bi-directional attention are the same.
                    ignore_causal_mask = True

                # Unfortunately, for query_length > 1 and key_value_length != query_length, we cannot generally ignore the attention mask, as SDPA causal mask generation
                # may be wrong. We will set `is_causal=False` in SDPA and rely on Transformers attention_mask instead, hence not setting it to None here.
                # Reference: https://github.com/pytorch/pytorch/issues/108108
                # TODO: maybe revisit this with https://github.com/pytorch/pytorch/pull/114823 in PyTorch 2.3.

        return ignore_causal_mask


def get_sequence_indices(attention_mask: torch.Tensor) -> tuple[int, int, int]:
    sequence_indices = []
    for i, a in enumerate(attention_mask):
        current_sid = a[0]
        s = 0
        e = -1
        for sid in a:
            if sid == current_sid:
                e += 1
            else:
                sequence_indices.append((i, s, e))
                e += 1
                s = e
                current_sid = sid
        else:
            sequence_indices.append((i, s, e))
    return sequence_indices


def update_4d_causal_mask_for_packed_sequence(
    causal_mask: torch.Tensor,
    sequence_indices: tuple[int, int, int],
    inplace: bool = False
) -> torch.Tensor:
    causal_mask = causal_mask if inplace else causal_mask.clone()
    min_dtype = torch.finfo(causal_mask.dtype).min
    for i, s, e in sequence_indices:
        causal_mask[i, :, s:e + 1, :s] = min_dtype
    return causal_mask


def prepare_4d_causal_attention_mask(
    attention_mask: Optional[torch.Tensor],
    input_shape: Union[torch.Size, Tuple, List],
    inputs_embeds: torch.Tensor,
    past_key_values_length: int,
    sliding_window: Optional[int] = None,
):
    """
    Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`

    Args:
        attention_mask (`torch.Tensor` or `None`):
            A 2D attention mask of shape `(batch_size, key_value_length)`
        input_shape (`tuple(int)` or `list(int)` or `torch.Size`):
            The input shape should be a tuple that defines `(batch_size, query_length)`.
        inputs_embeds (`torch.Tensor`):
            The embedded inputs as a torch Tensor.
        past_key_values_length (`int`):
            The length of the key value cache.
        sliding_window (`int`, *optional*):
            If the model uses windowed attention, a sliding window should be passed.
    """

    sequence_indices = get_sequence_indices(attention_mask)
    attention_mask = attention_mask.masked_fill(attention_mask > 1, 1)

    attn_mask_converter = AttentionMaskConverter(is_causal=True, sliding_window=sliding_window)

    key_value_length = input_shape[-1] + past_key_values_length

    # 4d mask is passed through the layers
    if attention_mask is not None and len(attention_mask.shape) == 2:
        causal_mask = attn_mask_converter.to_4d(
            attention_mask, input_shape[-1], key_value_length=key_value_length, dtype=inputs_embeds.dtype
        )
    elif attention_mask is not None and len(attention_mask.shape) == 4:
        expected_shape = (input_shape[0], 1, input_shape[1], key_value_length)
        if tuple(attention_mask.shape) != expected_shape:
            raise ValueError(
                f"Incorrect 4D attention_mask shape: {tuple(attention_mask.shape)}; expected: {expected_shape}."
            )
        else:
            # if the 4D mask has correct shape - invert it and fill with negative infinity
            inverted_mask = 1.0 - attention_mask
            causal_mask = inverted_mask.masked_fill(
                inverted_mask.to(torch.bool), torch.finfo(inputs_embeds.dtype).min
            )
    else:
        causal_mask = attn_mask_converter.to_causal_4d(
            input_shape[0], input_shape[-1], key_value_length, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )

    causal_mask = update_4d_causal_mask_for_packed_sequence(causal_mask, sequence_indices)
    
    return causal_mask


def get_max_seqlen_in_batch(attention_mask: torch.Tensor):
    attention_mask = attention_mask.to(dtype=torch.int32)
    max_num = torch.max(attention_mask)
    # attention_mask: B x N
    counts = []
    for i in range(1, max_num + 1):
        counts.append(
            torch.sum(attention_mask == i, axis=-1)
        )  # shape: B, count length of data point maksed with i
    result = torch.stack(counts, axis=1)
    result = result.flatten()
    return result[result.nonzero()].squeeze(-1).to(dtype=torch.int32)


def _get_unpad_data(attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Retrieves indexing data required to repad unpadded (ragged) tensors.

    Arguments:
        attention_mask (`torch.Tensor`):
            Boolean or int tensor of shape (batch_size, sequence_length), 0 means not valid and non-zero means valid.

    Return:
        indices (`torch.Tensor`):
            The indices of non-masked tokens from the flattened input sequence.
        cu_seqlens (`torch.Tensor`):
            The cumulative sequence lengths, used to index into ragged (unpadded) tensors. `cu_seqlens` shape is (batch_size + 1,).
        max_seqlen_in_batch (`int`):
            Maximum sequence length in batch.
    """
    seqlens_in_batch = get_max_seqlen_in_batch(attention_mask)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def _upad_input(
    query_layer: torch.Tensor,
    key_layer: torch.Tensor,
    value_layer: torch.Tensor,
    attention_mask: torch.Tensor,
    query_length: int,
):
    """
    Unpads query, key, and values tensors, using a single dimension for all tokens even though they belong to different batches.

    This function is used instead of `flash_attn.bert_padding.unpad_input` in order to avoid the recomputation of the same intermediary
    tensors for query, key, value tensors.

    Arguments:
        query_layer (`torch.Tensor`):
            Query state with padding. Shape: (batch_size, query_length, num_heads, head_dim).
        key_layer (`torch.Tensor`):
            Key state with padding. Shape: (batch_size, kv_seq_len, num_key_value_heads, head_dim).
        value_layer (`torch.Tensor`):
            Value state with padding. Shape: (batch_size, kv_seq_len, num_key_value_heads, head_dim).
        attention_mask (`torch.Tensor`):
            Boolean or int tensor of shape (batch_size, sequence_length), 1 means valid and 0 means not valid.
        query_length (`int`):
            Target length.

    Return:
        query_layer (`torch.Tensor`):
            Query state without padding. Shape: (total_target_length, num_heads, head_dim).
        key_layer (`torch.Tensor`):
            Key state with padding. Shape: (total_source_length, num_key_value_heads, head_dim).
        value_layer (`torch.Tensor`):
            Value state with padding. Shape: (total_source_length, num_key_value_heads, head_dim).
        indices_q (`torch.Tensor`):
            The indices of non-masked tokens from the flattened input target sequence.
        (cu_seqlens_q, cu_seqlens_k) (`Tuple[int]`):
            The cumulative sequence lengths for the target (query) and source (key, value), used to index into ragged (unpadded) tensors. `cu_seqlens` shape is (batch_size + 1,).
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k) (`Tuple[int]`):
            Maximum sequence length in batch (`max_seqlen_in_batch_q` for the target sequence i.e. query, `max_seqlen_in_batch_k` for the source sequence i.e. key/value).
    """
    indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
    batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

    key_layer = index_first_axis(key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k)
    value_layer = index_first_axis(
        value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
    )
    if query_length == kv_seq_len:
        query_layer = index_first_axis(query_layer.reshape(batch_size * kv_seq_len, -1, head_dim), indices_k)
        cu_seqlens_q = cu_seqlens_k
        max_seqlen_in_batch_q = max_seqlen_in_batch_k
        indices_q = indices_k
    elif query_length == 1:
        max_seqlen_in_batch_q = 1
        cu_seqlens_q = torch.arange(
            batch_size + 1, dtype=torch.int32, device=query_layer.device
        )  # There is a memcpy here, that is very bad.
        indices_q = cu_seqlens_q[:-1]
        query_layer = query_layer.squeeze(1)
    else:
        # The -q_len: slice assumes left padding.
        attention_mask = attention_mask[:, -query_length:]
        query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

    return (
        query_layer,
        key_layer,
        value_layer,
        indices_q,
        (cu_seqlens_q, cu_seqlens_k),
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
    )


def prepare_fa2_from_position_ids(query, key, value, position_ids):
    """
    This function returns necessary arguments to call `flash_attn_varlen_func`.
    All three query, key, value states will be flattened.
    Cummulative lengths of each examples in the batch will be extracted from position_ids.

    NOTE: ideally cummulative lengths should be prepared at the data collator stage

    Arguments:
        query (`torch.Tensor`):
            Query state with padding. Shape: (batch_size, query_length, num_heads, head_dim).
        key (`torch.Tensor`):
            Key state with padding. Shape: (batch_size, kv_seq_len, num_key_value_heads, head_dim).
        value (`torch.Tensor`):
            Value state with padding. Shape: (batch_size, kv_seq_len, num_key_value_heads, head_dim).
        position_ids (`torch.Tensor`):
            Boolean or int tensor of shape (batch_size, sequence_length), 1 means valid and 0 means not valid.

    Return:
        query (`torch.Tensor`):
            Query state without padding. Shape: (total_target_length, num_heads, head_dim).
        key (`torch.Tensor`):
            Key state with padding. Shape: (total_source_length, num_key_value_heads, head_dim).
        value (`torch.Tensor`):
            Value state with padding. Shape: (total_source_length, num_key_value_heads, head_dim).
        indices_q (`torch.Tensor`):
            The indices of non-masked tokens from the flattened input target sequence.
        (cu_seqlens_q, cu_seqlens_k) (`Tuple[int]`):
            The cumulative sequence lengths for the target (query) and source (key, value), used to index into ragged (unpadded) tensors. `cu_seqlens` shape is (batch_size + 1,).
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k) (`Tuple[int]`):
            Maximum sequence length in batch (`max_seqlen_in_batch_q` for the target sequence i.e. query, `max_seqlen_in_batch_k` for the source sequence i.e. key/value).
    """
    query = query.view(-1, query.size(-2), query.size(-1))
    key = key.view(-1, key.size(-2), key.size(-1))
    value = value.view(-1, value.size(-2), value.size(-1))
    position_ids = position_ids.flatten()
    indices_q = torch.arange(position_ids.size(0), device=position_ids.device, dtype=torch.int32)

    cu_seq_lens = torch.cat(
        (
            indices_q[position_ids == 0],
            torch.tensor(position_ids.size(), device=position_ids.device, dtype=torch.int32),
        )
    )

    max_length = position_ids.max() + 1

    return (query, key, value, indices_q, (cu_seq_lens, cu_seq_lens), (max_length, max_length))


def flash_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor,
    query_length: int,
    is_causal: bool,
    dropout: float = 0.0,
    position_ids: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    sliding_window: Optional[int] = None,
    use_top_left_mask: bool = False,
    softcap: Optional[float] = None,
    deterministic: bool = None,
):
    """
    Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
    first unpad the input, then computes the attention scores and pad the final attention scores.

    Args:
        query_states (`torch.Tensor`):
            Input query states to be passed to Flash Attention API
        key_states (`torch.Tensor`):
            Input key states to be passed to Flash Attention API
        value_states (`torch.Tensor`):
            Input value states to be passed to Flash Attention API
        attention_mask (`torch.Tensor`):
            The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
            position of padding tokens and 1 for the position of non-padding tokens.
        dropout (`float`):
            Attention dropout
        softmax_scale (`float`, *optional*):
            The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        use_top_left_mask (`bool`, defaults to `False`):
            flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference.
        softcap (`float`, *optional*):
            Softcap for the attention logits, used e.g. in gemma2.
        deterministic (`bool`, *optional*):
            Determines if the deterministic option introduced in flash_attn>=2.4.1 is enabled.
    """
    if not use_top_left_mask:
        causal = is_causal
    else:
        # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__.
        causal = is_causal and query_length != 1

    # Assuming 4D tensors, key_states.shape[1] is the key/value sequence length (source length).
    use_sliding_windows = (
        _flash_supports_window_size and sliding_window is not None and key_states.shape[1] > sliding_window
    )
    flash_kwargs = {"window_size": (sliding_window, sliding_window)} if use_sliding_windows else {}

    if deterministic is None:
        deterministic = os.environ.get("FLASH_ATTENTION_DETERMINISTIC", "0") == "1"
    flash_kwargs["deterministic"] = deterministic

    if softcap is not None:
        flash_kwargs["softcap"] = softcap

    # Contains at least one padding token in the sequence
    if attention_mask is not None:
        batch_size = query_states.shape[0]
        query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = _upad_input(
            query_states, key_states, value_states, attention_mask, query_length
        )
        cu_seqlens_q, cu_seqlens_k = cu_seq_lens
        max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

        attn_output_unpad = flash_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_in_batch_q,
            max_seqlen_k=max_seqlen_in_batch_k,
            dropout_p=dropout,
            softmax_scale=softmax_scale,
            causal=causal,
            **flash_kwargs,
        )
        attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)

    # If position_ids is provided and check all examples do not contain only 1 sequence, If tensor in increasing
    # then we probably have one sequence, otherwise it is packed. Additionally check we are in pre-fill/training stage.
    # Use `flash_attn_varlen_func` to prevent cross-example attention and also allow padding free approach
    elif position_ids is not None and not (torch.diff(position_ids, dim=-1) >= 0).all() and query_length != 1:
        batch_size = query_states.size(0)
        query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = prepare_fa2_from_position_ids(
            query_states, key_states, value_states, position_ids
        )

        cu_seqlens_q, cu_seqlens_k = cu_seq_lens
        max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

        attn_output = flash_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_in_batch_q,
            max_seqlen_k=max_seqlen_in_batch_k,
            dropout_p=dropout,
            softmax_scale=softmax_scale,
            causal=causal,
            **flash_kwargs,
        )

        attn_output = attn_output.view(batch_size, -1, attn_output.size(-2), attn_output.size(-1))

    else:
        attn_output = flash_attn_func(
            query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal, **flash_kwargs
        )

    return attn_output
