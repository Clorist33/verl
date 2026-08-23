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
from dataclasses import dataclass, field
from typing import Any, Optional

from omegaconf import MISSING
from transformers import AutoConfig

from verl.base_config import BaseConfig
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.import_utils import import_external_libs
from verl.utils.model import get_generation_config, update_model_config

__all__ = ["HFModelConfig", "MtpConfig", "Eagle3Config"]


@dataclass
class MtpConfig(BaseConfig):
    """
    Configuration for MTP model.

    enable: Enable loading and saving of MTP parameters, but do not use them

    enable_train: Whether to enable using MTP parameters during training
    enable_rollout: Whether to enable using MTP parameters during rollout

    Training parameters:
        detach_encoder: Whether to detach encoder parameters during MTP training
        mtp_loss_scaling_factor: Loss scaling factor during MTP training

    vLLM rollout parameters:
        method: "mtp"
        num-speculative-tokens: 1

    SGLang rollout parameters:
        speculative-algorithm: EAGLE
        speculative-num-steps: 3
        speculative-eagle-topk: 1
        speculative-num-draft-tokens: 4
    """

    enable: bool = False
    enable_train: bool = False
    enable_rollout: bool = False

    detach_encoder: bool = False
    mtp_loss_scaling_factor: float = 0.1

    speculative_algorithm: str = "EAGLE"
    speculative_num_steps: int = 3
    speculative_eagle_topk: int = 1
    speculative_num_draft_tokens: int = 4

    method: str = "mtp"
    num_speculative_tokens: int = 1


@dataclass
class Eagle3Config(BaseConfig):
    """EAGLE3 online speculative-decoding / draft-model training config.

    4.0: Megatron training backend + draft-independent optimizer + optional
    vocab compression. See eagle3_开发方案设计4.0.html section 3 for field docs.
    """

    _mutable_fields = {"enable"}

    # ---- main switches ----
    enable: bool = False
    enable_train: bool = False
    enable_rollout: bool = False
    draft_model_path: str = ""
    loss_weight: float = 1.0
    # empty -> auto-detect from draft ckpt; runtime fallback formula (2, N//2, N-3)
    capture_layer_ids: list[int] = field(default_factory=list)
    method: str = "eagle3"
    num_speculative_tokens: int = 3

    # ---- draft independent optimizer hyperparams ----
    draft_optim_lr: float = 1e-4
    draft_optim_weight_decay: float = 0.01
    draft_optim_clip_grad: float = 1.0
    draft_optim_offload: bool = False
    # Activation-checkpoint the draft backbone (time-for-memory). Default False =
    # unchanged behavior. Set True to cut draft-path activation peak (P5-09).
    draft_forward_checkpoint: bool = False

    # ---- vocab compression switch ----
    enable_vocab_compression: bool = False
    draft_vocab_size: int = 0
    vocab_mapping_path: str = ""

    # ---- 4.0 Megatron backend ----
    backend: str = "megatron"  # training backend (this version fixes megatron)
    ttt_length: int = 1  # TTT unroll steps (1 = off; >1 = autoregressive)
    draft_pipeline_stage: str = "last"  # which PP stage hosts draft
    allow_pp_gt_1: bool = False  # first version gate: only PP=1 supported

    # ---- draft implementation choice ----
    use_megatron_draft: bool = False  # True = MegatronModule + TP support; False = nn.Module + DDP (default)
    # When True, draft uses TransformerBlock and supports TP cutting (saves ~2.2GB/card with TP=4)
    # When False, draft uses plain nn.Module (current stable implementation, no TP)
    # Draft TP degree (only used when use_megatron_draft=True). 0 = auto: follow policy TP,
    # capped down to a divisor of num_query_groups (QKV-fusion constraint). A smaller
    # explicit divisor builds a dedicated draft TP sub-group (PP=1/CP=1 only; Step 7 HW-pending).
    draft_tensor_parallel_size: int = 0

    def __post_init__(self):
        # Auto-derive enable from sub-switches to prevent footguns
        if self.enable_train or self.enable_rollout:
            self.enable = True

        # ---- rollout (generation-side) constraints (ported from 3.0) ----
        if self.enable_rollout and not self.draft_model_path:
            raise ValueError(
                "Eagle3Config: draft_model_path must be set when enable_rollout=True. "
                "EAGLE3 requires an external draft model checkpoint."
            )
        if self.enable_rollout and self.num_speculative_tokens <= 0:
            raise ValueError(
                f"Eagle3Config: num_speculative_tokens must be positive when enable_rollout=True, "
                f"got {self.num_speculative_tokens}"
            )
        if self.capture_layer_ids and len(self.capture_layer_ids) != 3:
            raise ValueError(
                f"Eagle3Config: capture_layer_ids must contain exactly 3 layer indices when specified, "
                f"got {len(self.capture_layer_ids)} indices: {self.capture_layer_ids}. "
                "Leave empty to auto-detect from the draft checkpoint."
            )
        if self.capture_layer_ids and any(i < 0 for i in self.capture_layer_ids):
            raise ValueError(
                f"Eagle3Config: capture_layer_ids must be non-negative, got {self.capture_layer_ids}"
            )
        if self.loss_weight < 0:
            raise ValueError(f"Eagle3Config: loss_weight must be non-negative, got {self.loss_weight}")

        # ---- 4.0 training-side (Megatron) constraints ----
        if self.backend not in ("megatron", "fsdp"):
            raise ValueError(f"Eagle3Config: backend must be 'megatron' or 'fsdp', got {self.backend!r}")
        if self.enable_train and self.backend != "megatron":
            raise ValueError(
                f"Eagle3Config: this version only implements the Megatron training backend, "
                f"got backend={self.backend!r}. Set backend='megatron' to train the draft."
            )
        if self.ttt_length < 1:
            raise ValueError(f"Eagle3Config: ttt_length must be >= 1 (1 disables TTT), got {self.ttt_length}")
        if self.draft_pipeline_stage != "last":
            # allow explicit non-negative integer stage index, otherwise reject
            if not (self.draft_pipeline_stage.lstrip("-").isdigit() and int(self.draft_pipeline_stage) >= 0):
                raise ValueError(
                    f"Eagle3Config: draft_pipeline_stage must be 'last' or a non-negative int string, "
                    f"got {self.draft_pipeline_stage!r}"
                )

        # draft independent optimizer hyperparams
        if self.enable_train:
            if self.draft_optim_lr <= 0:
                raise ValueError(f"Eagle3Config: draft_optim_lr must be positive, got {self.draft_optim_lr}")
            if self.draft_optim_weight_decay < 0:
                raise ValueError(
                    f"Eagle3Config: draft_optim_weight_decay must be non-negative, got {self.draft_optim_weight_decay}"
                )
            if self.draft_optim_clip_grad < 0:
                raise ValueError(
                    f"Eagle3Config: draft_optim_clip_grad must be non-negative, got {self.draft_optim_clip_grad}"
                )

        # vocab compression switch
        if self.enable_vocab_compression and self.draft_vocab_size <= 0:
            raise ValueError(
                f"Eagle3Config: draft_vocab_size must be positive when enable_vocab_compression=True, "
                f"got {self.draft_vocab_size}"
            )
        # NOTE: PP>1 vs allow_pp_gt_1 gate needs the Megatron parallel size, which is
        # not visible here; it is enforced at engine initialize() (P2). See design 3.④.


@dataclass
class HFModelConfig(BaseConfig):
    # note that we separate model_path, model_config_path and tokenizer_path in case they are different
    _mutable_fields = {
        "model_type",
        "hf_config_path",
        "tokenizer_path",
        "hf_config",
        "generation_config",
        "tokenizer",
        "processor",
        "local_path",
        "architectures",
        "local_hf_config_path",
        "local_tokenizer_path",
        "mtp",
        "eagle3",
    }

    path: str = MISSING
    local_path: Optional[str] = None
    hf_config_path: Optional[str] = None
    local_hf_config_path: Optional[str] = None
    tokenizer_path: Optional[str] = None
    local_tokenizer_path: Optional[str] = None

    # model type, e.g., "language_model", "value_model"
    model_type: str = "language_model"

    # whether to load tokenizer. This is useful when we only want to load model config
    load_tokenizer: bool = True

    hf_config: Any = None
    generation_config: Any = None
    tokenizer: Any = None
    processor: Any = None

    # whether to use shared memory
    use_shm: bool = False
    trust_remote_code: bool = False

    # custom chat template for the model
    custom_chat_template: Optional[str] = None

    external_lib: Any = None

    override_config: dict = field(default_factory=dict)

    enable_gradient_checkpointing: bool = True
    enable_activation_offload: bool = False

    use_remove_padding: bool = True

    # TODO: unify fsdp and megatron lora config
    # fsdp lora related. We may setup a separate config later
    lora_rank: int = 0
    lora_alpha: int = 16
    target_modules: Optional[Any] = "all-linear"  # allow both "all-linear" and ["q_proj","k_proj"]
    target_parameters: Optional[list[str]] = None  # for lora adapter on nn.Parameter

    exclude_modules: Optional[str] = None

    # megatron lora config
    lora: dict[str, Any] = field(default_factory=dict)

    # path to pre-trained LoRA adapter to load for continued training
    lora_adapter_path: Optional[str] = None
    use_liger: bool = False

    use_fused_kernels: bool = False
    fused_kernel_options: dict = field(default_factory=dict)

    # TiledMLP configuration for memory-efficient MLP computation
    tiled_mlp: dict = field(default_factory=lambda: {"enabled": False, "num_shards": 4})

    architectures: Optional[list[str]] = None

    mtp: MtpConfig = field(default_factory=MtpConfig)
    eagle3: Eagle3Config = field(default_factory=Eagle3Config)

    def __post_init__(self):
        import_external_libs(self.external_lib)

        if self.hf_config_path is None:
            self.hf_config_path = self.path
        if self.tokenizer_path is None:
            self.tokenizer_path = self.path

        self.local_path = copy_to_local(self.path, use_shm=self.use_shm)

        # construct tokenizer
        if self.load_tokenizer:
            self.local_tokenizer_path = copy_to_local(self.tokenizer_path, use_shm=self.use_shm)
            self.tokenizer = hf_tokenizer(self.local_tokenizer_path, trust_remote_code=self.trust_remote_code)
            self.processor = hf_processor(self.local_tokenizer_path, trust_remote_code=self.trust_remote_code)

        # For base models (e.g. Qwen3.5-2b-Base), the processor may not have a chat_template
        # while the tokenizer does. Sync it so that processor.apply_chat_template() works.
        if (
            self.processor is not None
            and not getattr(self.processor, "chat_template", None)
            and getattr(self.tokenizer, "chat_template", None)
        ):
            self.processor.chat_template = self.tokenizer.chat_template

        if self.custom_chat_template is not None:
            if self.processor is not None:
                self.processor.chat_template = self.custom_chat_template
            else:
                self.tokenizer.chat_template = self.custom_chat_template

        self.local_hf_config_path = copy_to_local(self.hf_config_path, use_shm=self.use_shm)
        self.generation_config = get_generation_config(
            self.local_hf_config_path, trust_remote_code=self.trust_remote_code
        )

        # construct hf_config
        attn_implementation = self.override_config.get("attn_implementation", "flash_attention_2")
        try:
            self.hf_config = AutoConfig.from_pretrained(
                self.local_hf_config_path,
                trust_remote_code=self.trust_remote_code,
                attn_implementation=attn_implementation,
            )
        except ValueError as error:
            lookup_error = error.__cause__ or error.__context__
            if not isinstance(lookup_error, KeyError) or lookup_error.args != ("deepseek_v4",):
                raise
            from vllm.transformers_utils.config import get_config

            self.hf_config = get_config(
                self.local_hf_config_path,
                trust_remote_code=self.trust_remote_code,
                attn_implementation=attn_implementation,
            )

        override_config_kwargs = {}

        if self.tokenizer is not None:
            override_config_kwargs.update(
                {
                    "bos_token_id": self.tokenizer.bos_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "pad_token_id": self.tokenizer.pad_token_id,
                }
            )

        # TODO: (vermouth1992). self.config.model in megatron differs from that of fsdp in the override_config.
        override_config = (
            self.override_config["model_config"] if "model_config" in self.override_config else self.override_config
        )
        override_config_kwargs.update(override_config)
        update_model_config(self.hf_config, override_config_kwargs=override_config_kwargs)

        self.share_embeddings_and_output_weights = getattr(self.hf_config, "tie_word_embeddings", False)

        # get model architectures
        self.architectures = getattr(self.hf_config, "architectures", None)
        assert self.architectures is not None and len(self.architectures) == 1, (
            "Expect only one architecture, got {}".format(self.architectures)
        )

        # per model patch
        if getattr(self.hf_config, "model_type", None) == "kimi_vl":
            self.hf_config.text_config.topk_method = "greedy"

        # When MTP is disabled, zero out MTP layer counts from hf_config so that
        # downstream engine/worker code does not need to handle each MTP field format
        # individually. Supports both DeepSeek-style (num_nextn_predict_layers) and
        # Qwen3.5-style (mtp_num_hidden_layers, possibly nested under text_config).
        if not self.mtp.enable:
            if hasattr(self.hf_config, "num_nextn_predict_layers"):
                self.hf_config.num_nextn_predict_layers = 0
            if hasattr(self.hf_config, "mtp_num_hidden_layers"):
                self.hf_config.mtp_num_hidden_layers = 0
            if hasattr(self.hf_config, "text_config") and hasattr(self.hf_config.text_config, "mtp_num_hidden_layers"):
                self.hf_config.text_config.mtp_num_hidden_layers = 0

        # EAGLE3 and MTP both patch GPTModel._postprocess during training; they cannot
        # be enabled together. See eagle3_开发方案设计4.0.html section 3 (__post_init__ rule ④).
        if self.eagle3.enable_train and self.mtp.enable:
            raise ValueError(
                "EAGLE3 draft training (model.eagle3.enable_train=True) is mutually exclusive "
                "with MTP (model.mtp.enable=True): both patch GPTModel._postprocess. "
                "Disable one of them."
            )

        # Ensure target_modules is a str or list[str] (only if not None)
        if self.target_modules is not None:
            if not isinstance(self.target_modules, (str | list)):
                raise TypeError(
                    "target_modules must be a string or a list of strings, "
                    f"but got {type(self.target_modules).__name__}"
                )
            if isinstance(self.target_modules, list):
                for x in self.target_modules:
                    if not isinstance(x, str):
                        raise TypeError(
                            f"All elements in target_modules list must be strings, but found {type(x).__name__}"
                        )

    def get_processor(self):
        return self.processor if self.processor is not None else self.tokenizer
