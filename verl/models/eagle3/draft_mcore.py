# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Self-contained EAGLE3 draft model (route 2, no modelopt / no Megatron TP).

Streamlined port of verl-SpeCo ``verl_speco/models/eagle/llama_eagle.py``
``LlamaForCausalLMEagle3``. Adaptation choices (design 5.0, approach A):

* Only the **sdpa** attention backend is kept (flash/flex dropped) -- most
  portable, NPU-friendly, no external deps.
* Only the plain Llama-3 style RoPE is kept (linear/NTK/YARN/multimodal
  variants dropped).
* ``pretraining_tp`` weight slicing and ``@torch.compile`` are dropped
  (NPU-safety / simplicity). Draft is a plain replicated ``nn.Module`` -- no
  Megatron tensor-parallel layers (first version does NOT use TP).

Submodule attribute names (``embed_tokens``, ``midlayer``, ``fc``, ``fc_norm``,
``norm``, ``lm_head``, and the decoder-layer internals) are preserved verbatim
so real EAGLE3 draft checkpoints (e.g. AngelSlim/Qwen3-*_eagle3) load cleanly.

The ``forward`` returns per-step ``logits`` / masks and does NOT compute loss;
the loss (soft CE against teacher = policy logits) lives in ``loss_mcore.py``.
The ``ttt_length`` argument is the TTT switch: ``1`` = single step (no KV
cache), ``>1`` = multi-step autoregressive unroll (design P2a vs P2b).
"""

import logging
import math
import os
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint  # activation checkpointing for the draft backbone
from transformers.activations import ACT2FN

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# ---------------------------------------------------------------------------
# config helpers (ported from verl-SpeCo base.py / llama_eagle.py)
# ---------------------------------------------------------------------------
def _get_config_value(config, key: str, default=None):
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def _normalize_layer_ids(layer_ids) -> Optional[list]:
    if layer_ids is None:
        return None
    if isinstance(layer_ids, int):
        return [int(layer_ids)]
    if isinstance(layer_ids, str):
        raw = layer_ids.strip()
        if not raw:
            return None
        if raw.startswith("["):
            import json

            layer_ids = json.loads(raw)
        else:
            layer_ids = [part.strip() for part in raw.split(",") if part.strip()]
    return [int(layer_id) for layer_id in list(layer_ids)]


def _eagle3_aux_layer_ids_from_config(config) -> Optional[list]:
    eagle_config = _get_config_value(config, "eagle_config", None)
    candidates = (
        _get_config_value(eagle_config, "target_hidden_layer_ids", None),
        _get_config_value(eagle_config, "eagle_aux_hidden_state_layer_ids", None),
        _get_config_value(config, "target_hidden_layer_ids", None),
        _get_config_value(config, "eagle_aux_hidden_state_layer_ids", None),
        _get_config_value(config, "target_layer_ids", None),
    )
    for layer_ids in candidates:
        normalized = _normalize_layer_ids(layer_ids)
        if normalized is not None:
            return normalized
    return None


def resolve_eagle3_num_aux_hidden_states(config) -> int:
    """Number of aux hidden states the draft consumes (default 3)."""
    num_aux_hidden_states = _get_config_value(config, "num_aux_hidden_states", None)
    if num_aux_hidden_states is None:
        layer_ids = _eagle3_aux_layer_ids_from_config(config)
        num_aux_hidden_states = len(layer_ids) if layer_ids else 3
    num_aux_hidden_states = int(num_aux_hidden_states)
    if num_aux_hidden_states <= 0:
        raise ValueError(f"EAGLE3 num_aux_hidden_states must be positive, got {num_aux_hidden_states}")
    return num_aux_hidden_states


# ---------------------------------------------------------------------------
# attention mask / rope / kv helpers
# ---------------------------------------------------------------------------
def _make_causal_mask(input_ids_shape, dtype, device, past_key_values_length: int = 0):
    """Causal mask [bsz, 1, tgt_len, tgt_len + past]."""
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)
    if past_key_values_length > 0:
        mask = torch.cat(
            [torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1
        )
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """Expand [bsz, seq_len] padding mask to [bsz, 1, tgt_len, src_len]."""
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask
    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


def prepare_decoder_attention_mask(attention_mask, input_shape, inputs_embeds, past_key_values_length):
    combined_attention_mask = None
    if input_shape[-1] > 1:
        combined_attention_mask = _make_causal_mask(
            input_shape, inputs_embeds.dtype, device=inputs_embeds.device,
            past_key_values_length=past_key_values_length,
        )
    if attention_mask is not None:
        expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
            inputs_embeds.device
        )
        combined_attention_mask = (
            expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
        )
    return combined_attention_mask


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """(b, kv_heads, slen, head_dim) -> (b, kv_heads * n_rep, slen, head_dim)."""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def _rotary_seq_len_for_position_ids(seq_len: int, position_ids: Optional[torch.Tensor]) -> int:
    if position_ids is None or position_ids.numel() == 0:
        return int(seq_len)
    return max(int(seq_len), int(position_ids.detach().max().item()) + 1)


# ---------------------------------------------------------------------------
# RoPE (Llama-3 style only) + RMSNorm
# ---------------------------------------------------------------------------
class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None,
                 scaling_factor=None, low_freq_factor=None, high_freq_factor=None, orig_max_position=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        # Llama3 style rotary embedding frequency scaling (optional)
        if all(v is not None for v in [scaling_factor, low_freq_factor, high_freq_factor, orig_max_position]):
            self.scaling_factor = scaling_factor
            self.low_freq_factor = low_freq_factor
            self.high_freq_factor = high_freq_factor
            self.orig_max_position = orig_max_position
            low_freq_wavelen = orig_max_position / low_freq_factor
            high_freq_wavelen = orig_max_position / high_freq_factor
            wave_len = 2 * math.pi / inv_freq
            if low_freq_factor != high_freq_factor:
                smooth = (orig_max_position / wave_len - low_freq_factor) / (high_freq_factor - low_freq_factor)
            else:
                smooth = 0
            inv_freq = torch.where(
                wave_len < high_freq_wavelen,
                inv_freq,
                torch.where(
                    wave_len > low_freq_wavelen,
                    inv_freq / self.scaling_factor,
                    (1 - smooth) * inv_freq / self.scaling_factor + smooth * inv_freq,
                ),
            )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings + 20, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def reset_inv_freq(self, device=None, dtype=torch.float32):
        """Rebuild inv_freq on the target device/dtype (needed on NPU)."""
        device = device if device is not None else self.inv_freq.device
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim))
        if all(getattr(self, a, None) is not None
               for a in ("scaling_factor", "low_freq_factor", "high_freq_factor", "orig_max_position")):
            low_freq_wavelen = self.orig_max_position / self.low_freq_factor
            high_freq_wavelen = self.orig_max_position / self.high_freq_factor
            wave_len = 2 * math.pi / inv_freq
            if self.low_freq_factor != self.high_freq_factor:
                smooth = (self.orig_max_position / wave_len - self.low_freq_factor) / (
                    self.high_freq_factor - self.low_freq_factor
                )
            else:
                smooth = 0
            inv_freq = torch.where(
                wave_len < high_freq_wavelen, inv_freq,
                torch.where(wave_len > low_freq_wavelen, inv_freq / self.scaling_factor,
                            (1 - smooth) * inv_freq / self.scaling_factor + smooth * inv_freq),
            )
        self.register_buffer("inv_freq", inv_freq.to(torch.float32), persistent=False)
        self._set_cos_sin_cache(seq_len=self.max_seq_len_cached, device=device, dtype=dtype)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len and seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


# ---------------------------------------------------------------------------
# Attention (sdpa single-step + manual TTT cache multi-step) -- no TP
# ---------------------------------------------------------------------------
class LlamaAttention(nn.Module):
    """Multi-headed attention. q/k/v take 2*hidden (input_emb ++ hidden concat)."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings

        self.q_proj = nn.Linear(self.hidden_size * 2, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size * 2, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size * 2, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self._init_rope()

    def _init_rope(self):
        # Only plain / Llama-3 style RoPE kept (approach A). rope_scaling of type
        # "llama3" is honored; other scaling types fall back to plain RoPE.
        rope_scaling = getattr(self.config, "rope_scaling", None)

        def rope_get(key, default=None):
            if rope_scaling is None:
                return default
            if isinstance(rope_scaling, dict):
                return rope_scaling.get(key, default)
            return getattr(rope_scaling, key, default)

        self.rotary_emb = LlamaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=getattr(self.config, "rope_theta", 10000),
            scaling_factor=rope_get("factor"),
            low_freq_factor=rope_get("low_freq_factor"),
            high_freq_factor=rope_get("high_freq_factor"),
            orig_max_position=rope_get("original_max_position_embeddings"),
        )

    def forward(self, hidden_states, cache_hidden: Optional[List] = None, attention_mask=None,
                position_ids=None, past_key_values=None, output_attentions=False, use_cache=False):
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        if cache_hidden is None:
            # ---- single-step (ttt_length == 1): standard sdpa ----
            rotary_seq_len = _rotary_seq_len_for_position_ids(q_len, position_ids)
            cos, sin = self.rotary_emb(query_states, seq_len=rotary_seq_len)
            cos, sin = cos.to(query_states.device), sin.to(query_states.device)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states, key_states, value_states,
                attn_mask=attention_mask, is_causal=attention_mask is None, dropout_p=0.0,
            )
        else:
            # ---- multi-step (ttt_length > 1): manual attention over TTT cache ----
            lck = len(cache_hidden[0])
            shifted_position_ids = position_ids + lck
            rotary_seq_len = _rotary_seq_len_for_position_ids(q_len + lck, shifted_position_ids)
            cos, sin = self.rotary_emb(query_states, seq_len=rotary_seq_len)
            cos, sin = cos.to(query_states.device), sin.to(query_states.device)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, shifted_position_ids)
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            cache_hidden[0] = cache_hidden[0] + [key_states]
            cache_hidden[1] = cache_hidden[1] + [value_states]
            cache_k, cache_v = cache_hidden[0], cache_hidden[1]
            k0, v0 = cache_k[0], cache_v[0]

            attn_weights = torch.matmul(query_states, k0.transpose(2, 3)) / math.sqrt(self.head_dim)
            lck = len(cache_k)
            attn_weights = attn_weights + attention_mask
            for i in range(1, lck):
                ki = cache_k[i]
                attn_weightsi = (query_states * ki).sum(-1) / math.sqrt(self.head_dim)
                attn_weights = torch.cat((attn_weights, attn_weightsi[..., None]), dim=-1)
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights0 = attn_weights[..., :q_len]
            attn_output = torch.matmul(attn_weights0, v0)
            for i in range(1, lck):
                vi = cache_v[i]
                attn_weightsi = attn_weights[..., q_len + i - 1]
                attn_output = attn_output + attn_weightsi[..., None] * vi

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.head_dim * self.num_heads)
        return self.o_proj(attn_output)


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class LlamaDecoderLayer(nn.Module):
    """One EAGLE3 draft decoder layer. Concatenates (input_emb ++ hidden) before attn."""

    def __init__(self, config, attention_backend: str = "sdpa"):
        super().__init__()
        self.hidden_size = config.hidden_size
        if attention_backend != "sdpa":
            raise ValueError(
                f"draft_mcore (approach A) only supports attention_backend='sdpa', got {attention_backend!r}"
            )
        self.self_attn = LlamaAttention(config=config)
        self.attention_backend = attention_backend
        self.mlp = LlamaMLP(config)
        self.hidden_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_emb, hidden_states, cache_hidden=None, attention_mask=None,
                position_ids=None, past_key_values=None, output_attentions=False, use_cache=False):
        residual = hidden_states
        hidden_states = self.hidden_norm(hidden_states)
        input_emb = self.input_layernorm(input_emb)
        hidden_states = torch.cat((input_emb, hidden_states), dim=-1)
        hidden_states = self.self_attn(
            cache_hidden=cache_hidden, hidden_states=hidden_states, attention_mask=attention_mask,
            position_ids=position_ids, past_key_values=past_key_values,
            output_attentions=output_attentions, use_cache=use_cache,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


# ---------------------------------------------------------------------------
# EAGLE3 draft model
# ---------------------------------------------------------------------------
class LlamaForCausalLMEagle3(nn.Module):
    """Self-written EAGLE3 draft (plain nn.Module, no Megatron TP).

    Structure mirrors verl-SpeCo ``LlamaForCausalLMEagle3``:
    ``embed_tokens`` + ``fc`` (fuse ``num_aux`` aux hidden) + optional
    ``fc_norm`` + one ``midlayer`` decoder + ``norm`` + small-vocab ``lm_head``,
    plus ``t2d``/``d2t`` vocab-compression buffers.
    """

    _no_split_modules = ["LlamaDecoderLayer"]

    def __init__(self, config, attention_backend: str = "sdpa") -> None:
        super().__init__()
        self.config = config

        self.vocab_size = config.vocab_size
        self.draft_vocab_size = getattr(config, "draft_vocab_size", config.vocab_size)
        self.target_hidden_size = getattr(config, "target_hidden_size", config.hidden_size)
        self.num_aux_hidden_states = resolve_eagle3_num_aux_hidden_states(config)

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, getattr(config, "pad_token_id", None))
        self.midlayer = LlamaDecoderLayer(config, attention_backend=attention_backend)
        self.fc = nn.Linear(self.target_hidden_size * self.num_aux_hidden_states, config.hidden_size, bias=False)
        if getattr(config, "fc_norm", False):
            self.fc_norm = nn.ModuleList(
                [LlamaRMSNorm(self.target_hidden_size, eps=config.rms_norm_eps)
                 for _ in range(self.num_aux_hidden_states)]
            )
        else:
            self.fc_norm = None
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, self.draft_vocab_size, bias=False)

        # vocab-compression buffers (default identity; overwritten by
        # load_vocab_mapping when the draft ckpt ships its own t2d/d2t).
        t2d = torch.zeros(self.vocab_size, dtype=torch.bool)
        t2d[: self.draft_vocab_size] = True
        d2t = torch.arange(self.draft_vocab_size, dtype=torch.int64)
        self.register_buffer("t2d", t2d)
        self.register_buffer("d2t", d2t)
        self.vocab_mapping_loaded = False

        # Activation checkpointing on the draft backbone (time-for-memory trade).
        # Default False = byte-identical to the non-checkpointed path (reversible).
        # engine_support.setup toggles this from eagle3_cfg.draft_forward_checkpoint.
        # Only takes effect under ttt_length==1 + training (see forward()).
        self._use_forward_checkpoint = False

    # ---- vocab mapping (ported from verl-SpeCo base.py:131) ----
    def load_vocab_mapping(self, file_path: str) -> None:
        assert hasattr(self, "t2d") and hasattr(self, "d2t"), "t2d/d2t buffers missing"
        vocab_mapping = torch.load(file_path, map_location=self.t2d.device)
        t2d = vocab_mapping["t2d"].to(device=self.t2d.device, dtype=self.t2d.dtype)
        d2t = vocab_mapping["d2t"].to(device=self.d2t.device, dtype=self.d2t.dtype)
        if t2d.shape != self.t2d.shape:
            raise ValueError(f"Expected t2d shape {tuple(self.t2d.shape)}, got {tuple(t2d.shape)}")
        if d2t.shape != self.d2t.shape:
            raise ValueError(f"Expected d2t shape {tuple(self.d2t.shape)}, got {tuple(d2t.shape)}")
        self.t2d.copy_(t2d)
        self.d2t.copy_(d2t)
        self.vocab_mapping_loaded = True

    def reset_rope_buffers(self, dtype=torch.float32) -> int:
        """Rebuild RoPE inv_freq buffers on the current device (needed on NPU)."""
        reset_count = 0
        device = next(self.parameters()).device
        for module in self.modules():
            reset_inv_freq = getattr(module, "reset_inv_freq", None)
            if module is not self and callable(reset_inv_freq):
                reset_inv_freq(device=device, dtype=dtype)
                reset_count += 1
        return reset_count

    # ---- building blocks used by forward ----
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def project_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        expected = self.target_hidden_size * self.num_aux_hidden_states
        if hidden_states.size(-1) != expected:
            raise ValueError(
                f"EAGLE3 expects hidden_states last dim {expected}, got {hidden_states.size(-1)} "
                f"(target_hidden_size={self.target_hidden_size}, num_aux={self.num_aux_hidden_states})"
            )
        if self.fc_norm is not None:
            chunks = hidden_states.chunk(self.num_aux_hidden_states, dim=-1)
            hidden_states = torch.cat([norm(c) for norm, c in zip(self.fc_norm, chunks)], dim=-1)
        return self.fc(hidden_states)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.norm(hidden_states))

    def backbone(self, input_embeds, hidden_states, cache_hidden, attention_mask,
                 position_ids, past_key_values=None, output_attentions=False, use_cache=True):
        return self.midlayer(
            input_emb=input_embeds, hidden_states=hidden_states, cache_hidden=cache_hidden,
            attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values,
            output_attentions=output_attentions, use_cache=use_cache,
        )

    def _shift_right(self, x: torch.Tensor):
        """Right-shift for teacher forcing: drop first pos, pad 0 at end."""
        return torch.cat([x[:, 1:], torch.zeros_like(x[:, :1])], dim=1)

    def forward(self, input_ids, hidden_states, loss_mask, attention_mask=None,
                position_ids=None, past_key_value=None, ttt_length: int = 1):
        """Draft forward. Returns per-step logits + masks (NO loss here).

        Args:
            input_ids: (B, S) draft input tokens.
            hidden_states: (B, S, target_hidden_size * num_aux) concatenated
                aux hidden states captured from the policy (already detached).
            loss_mask: (B, S) valid-position mask.
            ttt_length: TTT switch. 1 = single step (no cache); >1 = multi-step
                autoregressive unroll (KV cache + per-step right shift).
        """
        if ttt_length == 1:
            cache_hidden = None
        else:
            cache_hidden = [[], []]

        batch_size, seq_length, _ = hidden_states.size()
        device = hidden_states.device

        current_hidden_states = self.project_hidden_states(hidden_states)

        if position_ids is None:
            past_length = 0
            if past_key_value is not None:
                past_length = past_key_value.get_usable_length(seq_length)
            position_ids = torch.arange(past_length, seq_length + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.bool, device=device)
        attention_mask = prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), hidden_states, 0
        )

        pad_id = getattr(self.config, "pad_token_id", None)
        if pad_id is None:
            pad_id = 0
        current_position_mask = (input_ids != pad_id).float().unsqueeze(-1)
        current_loss_mask = loss_mask
        current_input_ids = input_ids

        all_step_logits, all_step_loss_masks, all_step_position_masks = [], [], []

        # Activation checkpointing gate: only when explicitly enabled, single-step
        # (ttt_length==1 -> cache_hidden is None, backbone has no in-place cache
        # append to corrupt on recompute), and training (backward exists to save on).
        _ckpt = (
            getattr(self, "_use_forward_checkpoint", False)
            and ttt_length == 1
            and self.training
            and torch.is_grad_enabled()
        )

        for idx in range(ttt_length):
            is_last = idx == ttt_length - 1
            inputs_embeds = self.embed_input_ids(current_input_ids)
            if _ckpt:
                # Recompute the backbone in backward instead of storing its
                # activations -> cuts the draft-path activation peak. use_reentrant=False
                # is required so the detached aux `hidden_states` input (no grad) is
                # handled correctly and NPU autograd stays stable.
                current_hidden_states = torch.utils.checkpoint.checkpoint(
                    lambda emb, hs: self.backbone(
                        input_embeds=emb, hidden_states=hs,
                        cache_hidden=cache_hidden, attention_mask=attention_mask,
                        position_ids=position_ids, past_key_values=None,
                        output_attentions=False, use_cache=False,
                    ),
                    inputs_embeds, current_hidden_states,
                    use_reentrant=False,
                )
            else:
                current_hidden_states = self.backbone(
                    input_embeds=inputs_embeds, hidden_states=current_hidden_states,
                    cache_hidden=cache_hidden, attention_mask=attention_mask,
                    position_ids=position_ids, past_key_values=None,
                    output_attentions=False, use_cache=False,
                )
            logits = self.compute_logits(current_hidden_states)
            all_step_logits.append(logits)
            all_step_loss_masks.append(current_loss_mask)
            all_step_position_masks.append(current_position_mask)
            if not is_last:
                # Shift right one to emulate "predict the next token" for the next TTT step.
                current_input_ids = self._shift_right(current_input_ids)
                current_loss_mask = self._shift_right(current_loss_mask)
                current_position_mask = self._shift_right(current_position_mask)

        return {
            "logits": all_step_logits,
            "loss_masks": all_step_loss_masks,
            "position_masks": all_step_position_masks,
            "last_hidden_states": current_hidden_states,
        }

