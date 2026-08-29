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

"""Megatron-native EAGLE3 draft model with TP support.

Ported from /home/t00972278/lilac/verl/verl/workers/eagle/draft_model.py
and adapted to current verl EAGLE3 architecture.

Key differences from draft_mcore.py:
  - Base: MegatronModule (supports TP) instead of nn.Module
  - Decoder: Megatron TransformerBlock instead of HF LlamaDecoderLayer
  - Format: (S, B, H) instead of (B, S, H)
  - Output layer: borrows from GPTModel (TP-aware)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from megatron.core import mpu, tensor_parallel
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.models.common.embeddings import RotaryEmbedding


def _is_npu() -> bool:
    """Check if running on Ascend NPU (works even after MindSpeed torch.cuda redirect)."""
    try:
        return bool(torch.npu.is_available())
    except AttributeError:
        return False


def _has_accelerator() -> bool:
    """Check if CUDA or NPU is available."""
    return torch.cuda.is_available() or _is_npu()


def _get_current_device():
    """Get current accelerator device (NPU or CUDA)."""
    if _is_npu():
        return torch.npu.current_device()
    elif torch.cuda.is_available():
        return torch.cuda.current_device()
    else:
        return 'cpu'


class EagleModule(MegatronModule):
    """Core EAGLE3 module: fc + enorm + (2H-injected) TransformerBlock + TP output layer.

    Faithful port of the reference megatron-eagle EagleModule. The EAGLE3 layer-0
    attention consumes ``cat([enorm(emb), layernorm(hidden)], dim=-1)`` -> 2H, done
    via a STATEFUL forward-pre-hook on layer-0's qkv (whose in-features are widened
    to 2H). A last-layer forward-hook stashes the pre-final-norm hidden for TTT.
    """

    def __init__(
        self,
        config: TransformerConfig,
        target_hidden_size: int,
        num_aux_hidden_states: int,
        draft_vocab_size: int,
        pre_process: bool = True,
        post_process: bool = True,
        output_layer_weight: Optional[torch.Tensor] = None,
        tp_group=None,
        pg_collection=None,
    ):
        super().__init__(config)
        self.config = config
        self.target_hidden_size = target_hidden_size
        self.num_aux_hidden_states = num_aux_hidden_states
        self.draft_vocab_size = draft_vocab_size
        self.pre_process = pre_process
        self.post_process = post_process
        # Draft TP group. None -> the global TP group (draft TP == policy TP, the
        # validated default). A dedicated sub-group gives the draft an INDEPENDENT TP
        # degree (see engine_support._resolve_draft_tp_plan). self.tp_group is used for
        # the hand-built qkv/output layers + the logits gather; pg_collection is threaded
        # into TransformerBlock so attention/MLP shard on the same group.
        self.tp_group = tp_group
        self.pg_collection = pg_collection

        # fc: fuse num_aux aux hidden states (target_H*num_aux -> draft H). Plain
        # nn.Linear (not TP): aux comes from the policy at full dim.
        fc_input_dim = target_hidden_size * num_aux_hidden_states
        self.fc = nn.Linear(fc_input_dim, config.hidden_size, bias=False)

        # enorm: normalize the token embeddings before the 2H injection.
        self.enorm = nn.RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)

        # RoPE (TransformerBlock needs an explicit rotary_pos_emb). Pass
        # use_cpu_initialization so RotaryEmbedding doesn't call torch.cuda.current_device()
        # at construction (would crash on CPU, and is unnecessary anywhere).
        self.rotary_pos_emb = RotaryEmbedding(
            kv_channels=getattr(config, "kv_channels", config.hidden_size // config.num_attention_heads),
            rotary_percent=getattr(config, "rotary_percent", 1.0),
            rotary_base=getattr(config, "rotary_base", 10000),
            seq_len_interpolation_factor=getattr(config, "seq_len_interpolation_factor", None),
            rope_scaling=getattr(config, "rope_scaling", False),
            rope_scaling_factor=getattr(config, "rope_scaling_factor", 8.0),
            use_cpu_initialization=getattr(config, "use_cpu_initialization", not _has_accelerator()),
        )

        # decoder: single TransformerBlock (config.num_layers==1). pg_collection (when
        # given) pins attention/MLP TP layers to the draft's own group; None -> global.
        layer_spec = get_gpt_layer_local_spec(normalization="RMSNorm")
        block_kwargs = dict(config=config, spec=layer_spec,
                            pre_process=pre_process, post_process=post_process)
        if pg_collection is not None:
            block_kwargs["pg_collection"] = pg_collection
        self.decoder = TransformerBlock(**block_kwargs)

        # ---- EAGLE3 2H injection: widen layer-0 qkv, register hooks ----
        self._embeddings = None            # set each forward, consumed by the pre-hook
        self._next_hidden_states_input = None  # stashed by the last-layer hook (TTT)
        last_layer = self.decoder.layers[-1]
        last_layer.register_forward_hook(self._eagle3_layer_forward_hook)
        attn = self.decoder.layers[0].self_attention
        attn.register_forward_pre_hook(self._eagle3_attention_forward_pre_hook)
        # tp_group=None makes ColumnParallelLinear fall back to the global TP group.
        _qkv_kwargs = dict(
            config=attn.config, init_method=attn.config.init_method, gather_output=False,
            bias=attn.config.add_bias_linear or attn.config.add_qkv_bias,
            skip_bias_add=False, is_expert=False, tp_comm_buffer_name="qkv",
        )
        if tp_group is not None:
            _qkv_kwargs["tp_group"] = tp_group
        attn.linear_qkv = tensor_parallel.ColumnParallelLinear(
            attn.config.hidden_size * 2,  # 2H input (emb ++ hidden)
            attn.query_projection_size + 2 * attn.kv_projection_size,
            **_qkv_kwargs,
        )

        # output layer: TP-sharded lm_head over the DRAFT vocab (gather in forward).
        if post_process:
            _out_kwargs = dict(
                config=config, init_method=config.init_method, bias=False,
                skip_bias_add=False, gather_output=False, skip_weight_param_allocation=False,
            )
            if tp_group is not None:
                _out_kwargs["tp_group"] = tp_group
            self.eagle_output_layer = tensor_parallel.ColumnParallelLinear(
                config.hidden_size, draft_vocab_size, **_out_kwargs,
            )
        else:
            self.eagle_output_layer = None

    def _eagle3_layer_forward_hook(self, _module, _input, output):
        h = output.clone().detach() if isinstance(output, torch.Tensor) else output[0].clone().detach()
        self._next_hidden_states_input = h

    def _eagle3_attention_forward_pre_hook(self, _module, input_layernorm_output):
        if self._embeddings is None:
            raise ValueError("EagleModule attention pre-hook called before embeddings set")
        embeddings = self._embeddings
        self._embeddings = None
        # [S, B, H] ++ [S, B, H] -> [S, B, 2H]
        return torch.cat([embeddings, input_layernorm_output[0]], dim=-1)

    def compute_logits(self, hidden_states):
        """TP output layer -> gather draft-vocab shards -> full (S,B,draft_vocab).

        Gather on the DRAFT tp_group (self.tp_group); None falls back to the global
        group. tp size is taken from the group so it matches however the draft was sharded.
        """
        logits, _ = self.eagle_output_layer(hidden_states)  # (S,B,draft_vocab/TP)
        if self.tp_group is not None:
            tp_size = self.tp_group.size()
        else:
            tp_size = mpu.get_tensor_model_parallel_world_size()
        if tp_size > 1:
            logits = tensor_parallel.gather_from_tensor_model_parallel_region(
                logits, group=self.tp_group
            )
        return logits

    def forward(
        self,
        embeddings: torch.Tensor,     # (S, B, H) - input token embeddings
        hidden_states: torch.Tensor,  # (S, B, H) - already fc-projected aux hidden
        attention_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the 1-layer backbone with EAGLE3 2H injection.

        ``hidden_states`` is the fc-projected aux (S,B,H); ``embeddings`` is the token
        embeddings (S,B,H). enorm(embeddings) is stashed and the layer-0 qkv pre-hook
        concatenates it with the layernorm'd hidden -> 2H. Returns (final_hidden,
        next_hidden) where next_hidden is the pre-final-norm hidden (for TTT).
        """
        self._embeddings = self.enorm(embeddings)
        self._next_hidden_states_input = None
        if rotary_pos_emb is None:
            rotary_pos_emb = self.rotary_pos_emb(hidden_states.shape[0])

        hidden_states = self.decoder(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            rotary_pos_emb=rotary_pos_emb,
        )

        next_hidden = (
            hidden_states if self._next_hidden_states_input is None
            else self._next_hidden_states_input
        )
        self._next_hidden_states_input = None
        return hidden_states, next_hidden

    def _freeze_output_layer(self):
        if self.eagle_output_layer is not None:
            for p in self.eagle_output_layer.parameters():
                p.requires_grad_(False)

    def _unfreeze_output_layer(self):
        if self.eagle_output_layer is not None:
            for p in self.eagle_output_layer.parameters():
                p.requires_grad_(True)


class MegatronEagle3DraftModel(MegatronModule):
    """Top-level EAGLE3 draft model (Megatron-based).

    Wraps EagleModule and adds embedding layer + input processing.
    """

    def __init__(
        self,
        config: TransformerConfig,
        vocab_size: int,
        draft_vocab_size: int,
        target_hidden_size: int,
        num_aux_hidden_states: int = 1,
        pre_process: bool = True,
        post_process: bool = True,
        output_layer_weight: Optional[torch.Tensor] = None,
        tp_group=None,
        pg_collection=None,
    ):
        super().__init__(config)
        self.config = config
        self.vocab_size = vocab_size
        self.draft_vocab_size = draft_vocab_size
        self.target_hidden_size = target_hidden_size
        self.num_aux_hidden_states = num_aux_hidden_states
        self.pre_process = pre_process
        self.post_process = post_process
        # Draft TP group (None -> global). Threaded into EagleModule.
        self.tp_group = tp_group
        self.pg_collection = pg_collection

        # Embedding layer (not TP-parallelized in reference impl)
        # TODO: Consider using VocabParallelEmbedding for TP
        if pre_process:
            self.embed_tokens = nn.Embedding(vocab_size, config.hidden_size)
        else:
            self.embed_tokens = None

        # Core EAGLE module
        self.eagle_module = EagleModule(
            config=config,
            target_hidden_size=target_hidden_size,
            num_aux_hidden_states=num_aux_hidden_states,
            draft_vocab_size=draft_vocab_size,
            pre_process=pre_process,
            post_process=post_process,
            output_layer_weight=output_layer_weight,
            tp_group=tp_group,
            pg_collection=pg_collection,
        )

        # Vocab-compression buffers. Default identity so the patch's `draft.t2d`
        # (bool selector over full vocab -> draft vocab) is always a valid tensor;
        # overwritten by set_vocab_mapping / weight load when the ckpt ships its own.
        t2d = torch.zeros(vocab_size, dtype=torch.bool)
        t2d[:draft_vocab_size] = True
        d2t = torch.arange(draft_vocab_size, dtype=torch.int64)
        self.register_buffer("t2d", t2d, persistent=True)
        self.register_buffer("d2t", d2t, persistent=True)

    def set_vocab_mapping(self, t2d: Optional[torch.Tensor], d2t: Optional[torch.Tensor]):
        """Set teacher-to-draft and draft-to-teacher vocab mappings."""
        if t2d is not None:
            self.register_buffer("t2d", t2d, persistent=True)
        if d2t is not None:
            self.register_buffer("d2t", d2t, persistent=True)

    def _build_causal_mask(self, seq_len, batch, device):
        """Megatron causal mask: True = mask out (opposite of HF). NPU needs an
        explicit [B,1,S,S] mask (MindSpeed's None-branch injects an FA mask the
        local attention path can't broadcast)."""
        causal = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1
        )
        return causal.view(1, 1, seq_len, seq_len).expand(batch, 1, seq_len, seq_len)

    def forward(
        self,
        input_ids: torch.Tensor,       # (B, S) draft input tokens
        hidden_states: torch.Tensor,   # (B, S, target_H * num_aux) aux hidden from policy
        loss_mask: Optional[torch.Tensor] = None,   # (B, S)
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value=None,
        ttt_length: int = 1,
    ) -> dict:
        """Draft forward matching the nn.Module contract (eagle3_patch.py:228).

        Inputs are batch-first (B,S,*); internally transposed to Megatron seq-first
        (S,B,*). Returns per-step lists so compute_draft_loss can consume them:
          {"logits": [(B,S,draft_vocab)], "loss_masks": [...], "position_masks": [(B,S,1)], ...}
        Only ttt_length==1 is implemented (the P2a default); >1 raises.
        """
        if ttt_length != 1:
            raise NotImplementedError(
                "Megatron draft currently supports ttt_length==1 only (multi-step TTT "
                "with the layer-0 stateful pre-hook + KV cache is not yet ported)."
            )

        batch, seq_len, _ = hidden_states.shape
        device = hidden_states.device

        # batch-first (B,S,*) -> Megatron seq-first (S,B,*)
        ids_sbf = input_ids.transpose(0, 1).contiguous()            # (S, B)
        hidden_sbf = hidden_states.transpose(0, 1).contiguous()     # (S, B, H*num_aux)

        inputs_embeds = self.embed_tokens(ids_sbf)                  # (S, B, H)

        if attention_mask is None and _is_npu():
            attention_mask = self._build_causal_mask(seq_len, batch, device)

        # fc: fuse aux hidden (S,B,H*num_aux) -> (S,B,H). Applied ONCE here (the
        # reference applies fc at the top level, then EagleModule injects embeddings).
        hidden_sbf = self.eagle_module.fc(hidden_sbf)

        # RoPE positions. verl's THD forward passes position_ids=None unless MTP
        # training is on (model_forward.py:366), so the in-forward path lands on
        # offset=0 and behaves exactly as before. Deferred training harvests a
        # *window* out of the middle of a sequence and supplies its absolute
        # positions, so RoPE has to start where the window does -- verl-SpeCo does
        # the same, feeding hidden_positions + 1 and indexing its RoPE table by
        # them (eagle3_trainer_backend.py:820-822, llama_eagle.py:164).
        #
        # Attention scores only depend on relative offsets, so a shifted window is
        # self-consistent either way. What the offset buys is matching the phase
        # range the drafter meets at inference, where positions are absolute.
        rope_offset = 0
        if position_ids is not None and position_ids.numel() > 0:
            flat = position_ids.reshape(-1) if position_ids.dim() > 1 else position_ids
            rope_offset = int(flat[0].item())
        rotary_pos_emb = self.eagle_module.rotary_pos_emb(seq_len, offset=rope_offset)
        final_hidden, _ = self.eagle_module(
            embeddings=inputs_embeds,
            hidden_states=hidden_sbf,
            attention_mask=attention_mask,
            rotary_pos_emb=rotary_pos_emb,
        )

        logits_sbf = self.eagle_module.compute_logits(final_hidden)  # (S, B, draft_vocab)
        logits = logits_sbf.transpose(0, 1).contiguous()            # (B, S, draft_vocab)

        # position mask: valid (non-pad) draft positions, (B,S,1) as the loss expects.
        pad_id = getattr(self.config, "pad_token_id", None)
        if pad_id is None:
            pad_id = 0
        position_mask = (input_ids != pad_id).float().unsqueeze(-1)  # (B, S, 1)

        return {
            "logits": [logits],
            "loss_masks": [loss_mask],
            "position_masks": [position_mask],
            "last_hidden_states": final_hidden,
        }

    def freeze_output_layer(self):
        self.eagle_module._freeze_output_layer()

    def unfreeze_output_layer(self):
        self.eagle_module._unfreeze_output_layer()
