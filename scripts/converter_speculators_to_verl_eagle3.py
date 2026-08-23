# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""Convert a vLLM ``speculators``-format EAGLE3 draft checkpoint into the
layout verl's ``LlamaForCausalLMEagle3`` (verl/models/eagle3/draft_mcore.py)
expects to load.

Why this is needed
------------------
Public EAGLE3 drafts (e.g. ``Qwen3-32B-speculator.eagle3``) ship in the
``speculators`` library format:

  * ``config.json`` has NO top-level ``model_type`` -- the real layer config is
    nested under ``transformer_layer_config`` and the top level carries
    ``architectures=["Eagle3Speculator"]`` + ``speculators_config``.
    verl's ``build_draft_hf_config`` calls ``AutoConfig.from_pretrained`` which
    hard-fails on such a config ("Unrecognized model ... should have a
    model_type key").
  * The single decoder layer is named ``layers.0.*`` whereas verl's draft names
    it ``midlayer.*``. Every other tensor (embed_tokens/fc/norm/lm_head + the
    t2d/d2t vocab-mapping buffers) already matches by name.

This script rewrites both so the result is a plain local dir that
``build_draft_module`` loads with a full key match (zero random-init).

Usage
-----
    python scripts/converter_speculators_to_verl_eagle3.py \
        --src /mnt/share/weights/Qwen3-32B-speculator.eagle3 \
        --dst /mnt/share/weights/Qwen3-32B-eagle3-verl \
        --target-hidden-size 5120     # policy (Qwen3-32B) hidden_size

``--target-hidden-size`` is optional; verl fills it in at load time from the
policy config, but writing it here keeps the draft dir self-describing.
"""

import argparse
import json
import os

import torch
from safetensors.torch import load_file, save_file

# Tensors that keep their name unchanged between the two layouts.
_PASSTHROUGH = {
    "embed_tokens.weight",
    "fc.weight",
    "norm.weight",
    "lm_head.weight",
    "t2d",  # vocab-mapping buffers shipped by the draft ckpt
    "d2t",
}


def _remap_key(k: str) -> str:
    """speculators key -> verl LlamaForCausalLMEagle3 key.

    The only structural rename is the single decoder layer:
        layers.0.<rest>  ->  midlayer.<rest>
    Everything else is passed through untouched.
    """
    if k.startswith("layers.0."):
        return "midlayer." + k[len("layers.0."):]
    return k


def build_verl_config(spec_cfg: dict, target_hidden_size: int | None) -> dict:
    """Flatten a speculators config.json into a standard llama config.json
    that ``AutoConfig.from_pretrained`` can load."""
    tlc = spec_cfg["transformer_layer_config"]
    cfg = {
        "model_type": "llama",  # <- the missing key that makes AutoConfig work
        "architectures": ["LlamaForCausalLMEagle3"],
        "hidden_size": tlc["hidden_size"],
        "intermediate_size": tlc["intermediate_size"],
        "num_attention_heads": tlc["num_attention_heads"],
        "num_key_value_heads": tlc["num_key_value_heads"],
        "num_hidden_layers": 1,  # EAGLE3 draft is a single decoder layer
        "vocab_size": tlc["vocab_size"],
        "rms_norm_eps": tlc["rms_norm_eps"],
        "rope_theta": tlc["rope_theta"],
        "max_position_embeddings": tlc["max_position_embeddings"],
        "hidden_act": tlc.get("hidden_act", "silu"),
        "attention_bias": tlc.get("attention_bias", False),
        "attention_dropout": tlc.get("attention_dropout", 0.0),
        "initializer_range": tlc.get("initializer_range", 0.02),
        "torch_dtype": spec_cfg.get("torch_dtype", "bfloat16"),
        # EAGLE3-specific fields verl's draft reads:
        "draft_vocab_size": spec_cfg.get("draft_vocab_size", tlc["vocab_size"]),
        "norm_before_residual": spec_cfg.get("norm_before_residual", True),
    }
    if tlc.get("head_dim") is not None:
        cfg["head_dim"] = tlc["head_dim"]
    # target_hidden_size = policy hidden the draft fuses aux states from.
    # verl overrides this at load time, but record it for a self-describing dir.
    ths = target_hidden_size if target_hidden_size is not None else spec_cfg.get("target_hidden_size")
    if ths is not None:
        cfg["target_hidden_size"] = ths
    return cfg


def convert(src: str, dst: str, target_hidden_size: int | None) -> None:
    import glob

    if not os.path.isdir(src):
        raise NotADirectoryError(f"--src is not a directory: {src}")
    os.makedirs(dst, exist_ok=True)

    # 1) config.json: speculators (nested) -> flat llama config
    with open(os.path.join(src, "config.json")) as f:
        spec_cfg = json.load(f)
    verl_cfg = build_verl_config(spec_cfg, target_hidden_size)
    with open(os.path.join(dst, "config.json"), "w") as f:
        json.dump(verl_cfg, f, indent=2)
    print(f"[config] wrote {os.path.join(dst, 'config.json')} (model_type=llama, "
          f"hidden={verl_cfg['hidden_size']}, draft_vocab={verl_cfg['draft_vocab_size']})")

    # 2) weights: load every shard, remap keys, save as a single safetensors
    state = {}
    shards = sorted(glob.glob(os.path.join(src, "*.safetensors")))
    if not shards:
        raise FileNotFoundError(f"no *.safetensors found under {src}")
    for shard in shards:
        state.update(load_file(shard))

    remapped = {}
    renamed = 0
    for k, v in state.items():
        nk = _remap_key(k)
        if nk != k:
            renamed += 1
        # contiguous + keep dtype; save_file requires contiguous tensors
        remapped[nk] = v.contiguous()

    out_weights = os.path.join(dst, "model.safetensors")
    save_file(remapped, out_weights, metadata={"format": "pt"})
    print(f"[weights] {len(state)} tensors -> {out_weights} ({renamed} layer keys renamed layers.0.*->midlayer.*)")

    # 3) copy tokenizer files if present (harmless, keeps dir self-contained)
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json",
                 "merges.txt", "special_tokens_map.json", "generation_config.json"):
        sp = os.path.join(src, name)
        if os.path.isfile(sp):
            import shutil

            shutil.copy2(sp, os.path.join(dst, name))
            print(f"[copy] {name}")

    print(f"\nDone. Point eagle3.draft_model_path at: {dst}")


def _verify(dst: str, target_hidden_size: int | None) -> None:
    """Best-effort load check: rebuild verl draft from dst and confirm the
    state_dict loads with zero missing/unexpected keys."""
    try:
        from transformers import AutoConfig

        from verl.models.eagle3.draft_mcore import LlamaForCausalLMEagle3
    except Exception as e:
        print(f"[verify] skipped (import failed: {e})")
        return
    cfg = AutoConfig.from_pretrained(dst, trust_remote_code=True)
    if getattr(cfg, "target_hidden_size", None) is None and target_hidden_size is not None:
        cfg.target_hidden_size = target_hidden_size
    draft = LlamaForCausalLMEagle3(cfg, attention_backend="sdpa")
    state = load_file(os.path.join(dst, "model.safetensors"))
    missing, unexpected = draft.load_state_dict(state, strict=False)
    print(f"[verify] load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print(f"[verify]   missing: {sorted(missing)}")
    if unexpected:
        print(f"[verify]   unexpected: {sorted(unexpected)}")
    if not missing and not unexpected:
        print("[verify] OK -- full key match, draft loads pretrained weights (no random init).")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="speculators-format draft dir (has speculators_config)")
    ap.add_argument("--dst", required=True, help="output dir for verl-format draft")
    ap.add_argument("--target-hidden-size", type=int, default=None,
                    help="policy hidden_size (e.g. 5120 for Qwen3-32B); optional, verl fills at load")
    ap.add_argument("--no-verify", action="store_true", help="skip the post-conversion load check")
    args = ap.parse_args()

    convert(args.src, args.dst, args.target_hidden_size)
    if not args.no_verify:
        _verify(args.dst, args.target_hidden_size)


if __name__ == "__main__":
    main()
