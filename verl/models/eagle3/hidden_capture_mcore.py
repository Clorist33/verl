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
"""EAGLE3 aux hidden-state capture (Megatron, PP=1 first version).

The draft consumes 3 aux hidden states (low/mid/high layer) from the policy.
Mid-layer hidden is gone once it flows past, so it must be grabbed **in flight**
via forward hooks on the policy decoder layers -- it cannot be recovered from
``_postprocess`` (which only sees the final hidden).

Captured hidden is **detached** immediately: the draft's gradient must NOT flow
back into the policy (independent optimizer). This is the first of two gradient
isolation gates (the second being the separate draft optimizer / backward).

Reference: NeMo-RL ``hidden_capture.py`` (``feat/eagle3-online-specdec``) and
verl-SpeCo. First version is PP=1, so all 3 layers live on one rank and are
assembled locally; PP>1 cross-stage gather is deferred to P4.
"""

import logging
import os
from typing import List, Optional

import torch

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

try:
    from megatron.core.models.gpt.gpt_model import GPTModel
except ImportError:  # pragma: no cover - megatron always present in training env
    GPTModel = None

try:
    from megatron.core.utils import unwrap_model
except ImportError:  # pragma: no cover
    try:
        from verl.utils.megatron_utils import unwrap_model
    except ImportError:
        unwrap_model = None


def _resolve_gpt_model(model: torch.nn.Module):
    """Unwrap DDP/Float16Module etc. down to the GPTModel that owns decoder.layers."""
    m = unwrap_model(model) if unwrap_model is not None else model
    if GPTModel is not None and isinstance(m, GPTModel):
        return m
    # VLM-style: language_model is the GPTModel
    if hasattr(m, "language_model") and (GPTModel is None or isinstance(m.language_model, GPTModel)):
        return m.language_model
    # last resort: anything exposing decoder.layers
    if hasattr(m, "decoder") and hasattr(m.decoder, "layers"):
        return m
    raise TypeError(f"Eagle3HiddenCapture: cannot locate GPTModel/decoder.layers in {type(model).__name__}")


def get_eagle3_aux_hidden_state_layers(num_layers: int) -> List[int]:
    """Fallback formula for the 3 aux layers (low / mid / high).

    NeMo-RL reference: ``{1, num_layers // 2 - 1, num_layers - 4}``. Used ONLY
    when neither the draft ckpt (``eagle_aux_hidden_state_layer_ids``) nor the
    config (``capture_layer_ids``) provides explicit ids. Resolution priority is
    ckpt > config > this formula.
    """
    if num_layers < 4:
        # tiny models: just spread across what we have
        return sorted({0, max(0, num_layers // 2 - 1), max(0, num_layers - 1)})
    return [1, num_layers // 2 - 1, num_layers - 4]


def resolve_capture_layer_ids(
    num_layers: int,
    config_ids: Optional[List[int]] = None,
    ckpt_ids: Optional[List[int]] = None,
) -> List[int]:
    """Resolve which decoder layers to capture. Priority: ckpt > config > formula."""
    chosen = None
    if ckpt_ids:
        chosen = list(ckpt_ids)
        print("="*100)
        print("use ckpt_ids")
        print("="*100)
    elif config_ids:
        chosen = list(config_ids)
        print("="*100)
        print("use config_ids")
        print("="*100)
    else:
        chosen = get_eagle3_aux_hidden_state_layers(num_layers)
        print("="*100)
        print("use formula_ids")
        print("="*100)
        
    for i in chosen:
        if i < 0 or i >= num_layers:
            raise ValueError(
                f"Eagle3 capture layer id {i} out of range [0, {num_layers}); "
                f"resolved ids={chosen} (ckpt_ids={ckpt_ids}, config_ids={config_ids})"
            )
    return chosen


class Eagle3HiddenCapture:
    """Registers forward hooks on the policy decoder layers to capture 3 aux
    hidden states in flight, detaching them so no gradient flows to the policy.

    Usage (PP=1)::

        capture = Eagle3HiddenCapture(policy_model, capture_layer_ids=[1, 15, 28])
        capture.register()
        try:
            policy_forward(...)                # hooks fill the buffers
            aux = capture.get_captured(seqlen_first=True)   # (S, B, H*num_aux) or (B, S, H*num_aux)
        finally:
            capture.clear()                    # drop tensors (keep hooks for next step)
        # capture.remove()                     # detach hooks entirely when done

    Megatron ``TransformerLayer.forward`` returns ``(output, context)``; the hook
    grabs ``output[0]``. Hidden layout inside Megatron is ``(S, B, H)``
    (sequence-first); ``get_captured`` can return either layout.
    """

    def __init__(self, policy_model: torch.nn.Module, capture_layer_ids: List[int], detach: bool = True):
        self.gpt_model = _resolve_gpt_model(policy_model)
        self.layers = self.gpt_model.decoder.layers
        self.num_layers = len(self.layers)
        self.capture_layer_ids = list(capture_layer_ids)
        self.detach = detach
        self._captured = {}          # layer_id -> tensor
        self._handles = []           # hook handles
        self._row_index = None       # None -> keep every row; else (rows,) long
        for i in self.capture_layer_ids:
            if i < 0 or i >= self.num_layers:
                raise ValueError(
                    f"Eagle3HiddenCapture: layer id {i} out of range [0, {self.num_layers})"
                )

    def set_row_index(self, row_index):
        """Harvest only ``row_index`` sequence positions instead of the whole sequence.

        The full aux hidden is (S, B, H*num_aux); at S~6k, H=4096, num_aux=3 that is
        ~145 MB per sequence, which the deferred-training design cannot afford to
        stash (see 开发设计/串行训练/方案设计：SpeCo式采集与延后训练_v3.md §2).
        Slicing inside the hook keeps only the rows a
        :class:`~verl.models.eagle3.collect_plan.CollectPlan` asked for, so the
        stash cost scales with rows-used rather than sequence length.

        Slicing happens BEFORE ``detach()``, so the dropped rows become garbage as
        soon as the layer's output goes out of scope.

        NOTE: indices address the tensor the hook observes. Under sequence
        parallelism each rank holds only S/TP rows, so callers must pass
        rank-local indices (or gather first) -- this class does no coordinate
        translation and will raise if an index is out of range.

        Args:
            row_index: 1-D LongTensor of positions, or ``None`` to restore
                full-sequence capture.
        """
        if row_index is None:
            self._row_index = None
            return self
        if not isinstance(row_index, torch.Tensor):
            row_index = torch.as_tensor(row_index, dtype=torch.long)
        if row_index.dim() != 1:
            raise ValueError(f"row_index must be 1-D, got shape {tuple(row_index.shape)}")
        self._row_index = row_index.to(torch.long)
        return self

    @property
    def row_index(self):
        """Active row selection, or ``None`` when capturing the full sequence."""
        return self._row_index

    def _make_hook(self, layer_id: int):
        def hook(module, inputs, output):
            # Megatron TransformerLayer returns (hidden_states, context); plain
            # nn.Module may return a bare tensor.
            hs = output[0] if isinstance(output, (tuple, list)) else output
            if self._row_index is not None:
                # (S, B, H) -> (rows, B, H). Select before detach so the dropped
                # rows are freed with the layer output.
                idx = self._row_index
                if idx.device != hs.device:
                    idx = idx.to(hs.device)
                    self._row_index = idx
                if idx.numel() and int(idx.max()) >= hs.shape[0]:
                    raise IndexError(
                        f"Eagle3HiddenCapture: row index {int(idx.max())} out of range for "
                        f"layer {layer_id} output with {hs.shape[0]} sequence positions. "
                        "Under sequence_parallel each rank holds only S/TP rows -- pass "
                        "rank-local indices or gather before selecting."
                    )
                hs = hs.index_select(0, idx)
            self._captured[layer_id] = hs.detach() if self.detach else hs
        return hook


    def register(self):
        """Attach forward hooks. Idempotent-ish: call remove() before re-register."""
        if self._handles:
            return self
        for i in self.capture_layer_ids:
            h = self.layers[i].register_forward_hook(self._make_hook(i))
            self._handles.append(h)
        logger.info("Eagle3HiddenCapture: registered hooks on layers %s", self.capture_layer_ids)
        return self

    def get_captured(self, seqlen_first: bool = True) -> torch.Tensor:
        """Concatenate captured hidden states along the last dim in the order of
        ``capture_layer_ids`` -> ``(*, H * num_aux)``.

        seqlen_first: if True keep Megatron's (S, B, H) layout; if False return
        (B, S, H*num_aux) as the draft.forward expects.
        """
        missing = [i for i in self.capture_layer_ids if i not in self._captured]
        if missing:
            raise RuntimeError(
                f"Eagle3HiddenCapture: layers {missing} were not captured. "
                "Was the policy forward run inside register()? PP>1 not supported (P4)."
            )
        parts = [self._captured[i] for i in self.capture_layer_ids]
        cat = torch.cat(parts, dim=-1)   # concat on hidden dim
        if not seqlen_first and cat.dim() == 3:
            # (S, B, H*n) -> (B, S, H*n)
            cat = cat.transpose(0, 1).contiguous()
        return cat

    def clear(self):
        """Drop captured tensors (frees memory) but keep hooks for the next step."""
        self._captured = {}

    def remove(self):
        """Detach all hooks and drop captured tensors."""
        for h in self._handles:
            h.remove()
        self._handles = []
        self._captured = {}

    def __enter__(self):
        return self.register()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove()
        return False

