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

from dataclasses import is_dataclass
from typing import Any, Optional
import logging

from omegaconf import DictConfig, ListConfig, OmegaConf

logger = logging.getLogger(__name__)

__all__ = ["omega_conf_to_dataclass", "validate_config", "_validate_eagle3_serial_training_config"]


def _validate_eagle3_serial_training_config(config: DictConfig) -> None:
    """在训练启动前（worker初始化之前）验证 EAGLE3 串行训练配置

    串行模式要求：
    1. 必须设置 actor_training_steps
    2. total_training_steps 必须是 (k+1) 的整数倍（周期对齐）

    推导关系：
    - draft_training_steps = actor_training_steps // k
    - total_training_steps = actor_training_steps + draft_training_steps
    - 周期 = k+1（k 个 Actor 步 + 1 个 Draft 步）

    Args:
        config: Hydra 配置对象

    Raises:
        ValueError: 配置不合法时抛出，包含详细的错误说明和修复建议
    """
    # 检查是否启用串行训练
    eagle3_config = config.algorithm.get('eagle3', {})
    if not eagle3_config.get('enable_serial_training', False):
        return  # 未启用串行训练，跳过验证

    # 1. 获取配置参数
    actor_training_steps = config.trainer.get('actor_training_steps', None)
    k = eagle3_config.get('actor_steps_per_draft_step', 5)

    # 2. 验证：必须设置 actor_training_steps
    if actor_training_steps is None:
        raise ValueError(
            "[Serial Training] 串行训练模式必须设置 'actor_training_steps' 参数。\n"
            "示例配置：\n"
            "trainer:\n"
            "  actor_training_steps: 100  # Actor 训练步数（建议设为 k 的倍数）\n"
            "algorithm:\n"
            "  eagle3:\n"
            "    enable_serial_training: true\n"
            "    actor_steps_per_draft_step: 5  # k=5，周期为 k+1=6"
        )

    # 3. 计算推导参数
    draft_training_steps = actor_training_steps // k
    # v3：draft 搭 actor 步的车，不占独立 global step，所以总步数就是 actor 步数。
    total_training_steps = actor_training_steps

    # 4. 验证：total_training_steps 必须是 (k+1) 的整数倍（周期对齐）
    period = k + 1
    if total_training_steps % period != 0:
        # 计算最接近的合法值
        cycles_down = total_training_steps // period
        cycles_up = cycles_down + 1
        actor_down = cycles_down * k
        actor_up = cycles_up * k
        total_down = cycles_down * period
        total_up = cycles_up * period

        raise ValueError(
            f"[Serial Training] 配置不合法：\n"
            f"  actor_training_steps = {actor_training_steps}\n"
            f"  draft_training_steps = {draft_training_steps} (计算值: actor_training_steps // k)\n"
            f"  total_training_steps = {total_training_steps} (计算值: actor + draft)\n"
            f"  周期 (k+1) = {period}\n"
            f"\n"
            f"❌ 问题：total_training_steps ({total_training_steps}) 不是周期 ({period}) 的整数倍。\n"
            f"   这会导致最后几步的训练类型不符合预期（Actor/Draft 混乱）。\n"
            f"\n"
            f"✅ 建议修改 actor_training_steps 为以下值之一：\n"
            f"   - {actor_down}  → total={total_down} ({cycles_down} 个完整周期, {cycles_down} 个 Draft 步)\n"
            f"   - {actor_up}  → total={total_up} ({cycles_up} 个完整周期, {cycles_up} 个 Draft 步)\n"
            f"\n"
            f"💡 通用规则：actor_training_steps 设为 k 的倍数，即可保证周期对齐。"
        )

    # 5. 验证通过，输出日志
    num_cycles = total_training_steps // period
    logger.info("=" * 60)
    logger.info("[Serial Training] Configuration validated (before training):")
    logger.info(f"  actor_training_steps:       {actor_training_steps}")
    logger.info(f"  actor_steps_per_draft_step: {k}")
    logger.info(f"  draft_training_steps:       {draft_training_steps} (calculated)")
    logger.info(f"  total_training_steps:       {total_training_steps} (calculated)")
    logger.info(f"  training_ratio:             Actor:{actor_training_steps} / Draft:{draft_training_steps} = {k}:1")
    logger.info(f"  period (k+1):               {period} steps/cycle")
    logger.info(f"  num_cycles:                 {num_cycles} complete cycles")
    logger.info("=" * 60)


def omega_conf_to_dataclass(config: DictConfig | dict, dataclass_type: Optional[type[Any]] = None) -> Any:
    """
    Convert an OmegaConf DictConfig to a dataclass.

    Args:
        config: The OmegaConf DictConfig or dict to convert.
        dataclass_type: The dataclass type to convert to. When dataclass_type is None,
            the DictConfig must contain _target_ to be instantiated via hydra.instantiate API.

    Returns:
        The dataclass instance.
    """
    # Got an empty config
    if not config:
        return dataclass_type if dataclass_type is None else dataclass_type()
    # Got an object
    if not isinstance(config, DictConfig | ListConfig | dict | list):
        return config

    if dataclass_type is None:
        assert "_target_" in config, (
            "When dataclass_type is not provided, config must contain _target_. "
            "See trainer/config/ppo_trainer.yaml algorithm section for an example. "
            f"Got config: {config}"
        )
        from hydra.utils import instantiate

        return instantiate(config, _convert_="partial")

    if not is_dataclass(dataclass_type):
        raise ValueError(f"{dataclass_type} must be a dataclass")
    cfg = OmegaConf.create(config)  # in case it's a dict
    # pop _target_ to avoid hydra instantiate error, as most dataclass do not have _target_
    # Updated (vermouth1992) We add _target_ to BaseConfig so that it is compatible.
    # Otherwise, this code path can't support recursive instantiation.
    # if "_target_" in cfg:
    #     cfg.pop("_target_")
    cfg_from_dataclass = OmegaConf.structured(dataclass_type)
    # let cfg override the existing vals in `cfg_from_dataclass`
    cfg_merged = OmegaConf.merge(cfg_from_dataclass, cfg)
    # now convert to `dataclass_type`
    config_object = OmegaConf.to_object(cfg_merged)
    return config_object


def update_dict_with_config(dictionary: dict, config: DictConfig):
    for key in dictionary:
        if hasattr(config, key):
            dictionary[key] = getattr(config, key)


def validate_config(
    config: DictConfig,
    use_reference_policy: bool,
    use_critic: bool,
) -> None:
    """Validate an OmegaConf DictConfig.

    Args:
        config (DictConfig): The OmegaConf DictConfig to validate.
        use_reference_policy (bool): is ref policy needed
        use_critic (bool): is critic needed
    """
    # === EAGLE3 串行训练参数验证（在任何worker初始化前进行）===
    _validate_eagle3_serial_training_config(config)

    # number of GPUs total
    n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

    if not config.actor_rollout_ref.actor.use_dynamic_bsz:
        if config.actor_rollout_ref.actor.strategy == "megatron":
            model_parallel_size = (
                config.actor_rollout_ref.actor.megatron.tensor_model_parallel_size
                * config.actor_rollout_ref.actor.megatron.pipeline_model_parallel_size
            )
            assert (
                n_gpus % (model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size) == 0
            ), (
                f"n_gpus ({n_gpus}) must be divisible by model_parallel_size ({model_parallel_size}) times "
                f"context_parallel_size ({config.actor_rollout_ref.actor.megatron.context_parallel_size})"
            )
            megatron_dp = n_gpus // (
                model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size
            )
            minimal_bsz = megatron_dp * config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
        else:
            minimal_bsz = n_gpus

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % minimal_bsz == 0, (
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by minimal possible batch size "
            f"({minimal_bsz})"
        )

    # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
    # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
    def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
        """Validate mutually exclusive micro batch size configuration options.

        Ensures that users don't set both deprecated micro_batch_size and
        the new micro_batch_size_per_gpu parameters simultaneously.

        Args:
            mbs: Deprecated micro batch size parameter value.
            mbs_per_gpu: New micro batch size per GPU parameter value.
            name (str): Configuration section name for error messages.

        Raises:
            ValueError: If both parameters are set or neither is set.
        """
        settings = {
            "actor_rollout_ref.ref": "log_prob_micro_batch_size",
            "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
        }

        if name in settings:
            param = settings[name]
            param_per_gpu = f"{param}_per_gpu"

            if mbs is None and mbs_per_gpu is None:
                raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

            if mbs is not None and mbs_per_gpu is not None:
                raise ValueError(
                    f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove "
                    f"'{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
                )

    # Actor validation done in ActorConfig.__post_init__ and validate()
    actor_config = omega_conf_to_dataclass(config.actor_rollout_ref.actor)
    actor_config.validate(n_gpus, config.data.train_batch_size, config.actor_rollout_ref.model)

    if not config.actor_rollout_ref.actor.use_dynamic_bsz:
        if use_reference_policy:
            # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.ref",
            )

        #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
        check_mutually_exclusive(
            config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
            config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
            "actor_rollout_ref.rollout",
        )

    if config.algorithm.get("use_kl_in_reward", False) and config.actor_rollout_ref.actor.use_kl_loss:
        print("NOTICE: You have both enabled in-reward kl and kl loss.")

    # critic
    if use_critic:
        critic_config = omega_conf_to_dataclass(config.critic)
        critic_config.validate(n_gpus, config.data.train_batch_size)

    if config.data.get("val_batch_size", None) is not None:
        print(
            "WARNING: val_batch_size is deprecated."
            + " Validation datasets are sent to inference engines as a whole batch,"
            + " which will schedule the memory themselves."
        )

    # check eval config
    if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
        assert config.actor_rollout_ref.rollout.temperature > 0, (
            "validation gen temperature should be greater than 0 when enabling do_sample"
        )

    # check LoRA rank in vLLM
    lora_config = config.actor_rollout_ref.model.get("lora", {})
    lora_rank = lora_config.get("rank", 0)
    if lora_rank <= 0:
        lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
    if lora_config.get("merge", False):
        lora_rank = 0
    if lora_rank > 0 and config.actor_rollout_ref.rollout.name == "vllm":
        from verl.workers.rollout.vllm_rollout.utils import get_vllm_max_lora_rank

        get_vllm_max_lora_rank(lora_rank)

    print("[validate_config] All configuration checks passed successfully!")
