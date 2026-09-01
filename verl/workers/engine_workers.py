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
import functools
import logging
import os
from contextlib import nullcontext
from copy import deepcopy
from functools import partial
from itertools import chain
from typing import Optional

import psutil
import torch
from codetiming import Timer
from omegaconf import DictConfig, open_dict
from tensordict import NonTensorData, TensorDict
from torch.distributed.device_mesh import init_device_mesh

from verl.checkpoint_engine import CheckpointEngineRegistry
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.trainer.distillation import distillation_ppo_loss, is_distillation_enabled
from verl.utils import tensordict_utils as tu
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import get_device_name, get_torch_device, set_expandable_segments
from verl.utils.distributed import initialize_global_process_group_ray, set_numa_affinity
from verl.utils.flops_counter import FlopsCounter
from verl.utils.import_utils import import_external_libs
from verl.utils.memory_utils import aggressive_empty_cache
from verl.utils.metric.utils import Metric
from verl.utils.profiler import DistProfiler, DistProfilerExtension, ProfilerConfig, log_gpu_memory_usage
from verl.utils.py_functional import append_to_dict
from verl.utils.tensordict_utils import maybe_fix_3d_position_ids
from verl.utils.torch_functional import allgather_dict_into_dict
from verl.workers.config import (
    ActorConfig,
    DistillationConfig,
    HFModelConfig,
    MtpConfig,
    RolloutConfig,
    TrainingWorkerConfig,
)
from verl.workers.rollout.base import BaseRollout, get_rollout_class
from verl.workers.utils.losses import ppo_loss

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _eagle3_scalar_global_step(data) -> int:
    """Read ``global_steps`` off a batch as a plain int.

    ``global_steps`` is assigned per-sample during rollout
    (``trainer_base._generate_sequences``), so a batched TensorDict hands it back
    as a ``NonTensorStack`` -- one copy per row -- not a scalar. Both ``or`` and
    ``int()`` raise on that type (``RuntimeError: Converting a tensordict to
    boolean value is not permitted`` / ``TypeError``), which is what took down
    the first draft-training step of a run. Unwrap the stack and take the first
    entry: every row of a step carries the same step number.
    """
    value = tu.get_non_tensor_data(data, "global_steps", default=0)
    if hasattr(value, "tolist"):  # NonTensorStack, np.ndarray, torch.Tensor
        value = value.tolist()
    while isinstance(value, (list, tuple)):
        if not value:
            return 0
        value = value[0]
    return int(value) if value is not None else 0


def _with_routing_replay_flag(enabled: bool):
    """Decorator to set 'enable_routing_replay' flag on the data TensorDict."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, data: TensorDict, *args, **kwargs):
            if self.enable_routing_replay:
                tu.assign_non_tensor_data(data, "enable_routing_replay", enabled)
            return func(self, data, *args, **kwargs)

        return wrapper

    return decorator


class TrainingWorker(Worker, DistProfilerExtension):
    """
    TrainingWorker provides a Tinker-like API (https://thinkingmachines.ai/tinker/) as a RayWorkerGroup
    to a single controller. Currently, we only provide more coarse grained APIs,
    and do not provide exact APIs as Tinker does. But this can be added in the future.
    """

    def __init__(self, config: TrainingWorkerConfig):
        Worker.__init__(self)

        from verl.workers.engine import BaseEngine, EngineRegistry

        initialize_global_process_group_ray(timeout_second=None)

        set_numa_affinity()

        self.config = config
        self.model_config = self.config.model_config
        self.engine_config = self.config.engine_config
        self.optimizer_config = self.config.optimizer_config
        self.checkpoint_config = self.config.checkpoint_config
        self.device_name = get_device_name()

        if self.engine_config is None:
            assert self.optimizer_config is None
            if self.config.auto_select_engine_optim_fn is None:
                raise ValueError(
                    "engine_config is not provided and auto_select_engine_optim_fn is not set. "
                    "Cannot determine engine backend."
                )
            # Support automatically select engine backend given model config
            self.engine_config, self.optimizer_config = self.config.auto_select_engine_optim_fn(
                self.model_config, self.device_name
            )

        # we use the one defined in model
        # TODO: this is not elegant and should refactor later
        self.engine_config.use_remove_padding = self.model_config.get("use_remove_padding", False)
        self.engine_config.use_fused_kernels = self.model_config.get("use_fused_kernels", False)

        self.profiler_config = self.config.profiler_config
        if self.profiler_config is not None:
            self.profiler_tool_config = self.profiler_config.tool_config.get(self.profiler_config.tool, {})
        else:
            self.profiler_tool_config = None

        DistProfilerExtension.__init__(
            self,
            DistProfiler(
                rank=self.rank,
                config=self.profiler_config,
                tool_config=self.profiler_tool_config,
                # Embed the model role (e.g. language_model/value_model) in trace filenames
                # so standalone (e.g. SFT) traces are self-describing per process.
                save_file_prefix=getattr(self.config, "model_type", None),
            ),
        )

        self.model_config.model_type = self.config.model_type
        self.engine: BaseEngine = EngineRegistry.new(
            model_type=self.config.model_type,             # eagle3传入参数是"language_model"
            backend=self.engine_config.strategy,           # eagle3传入参数是megatron
            model_config=self.model_config,
            engine_config=self.engine_config,
            optimizer_config=self.optimizer_config,
            checkpoint_config=self.checkpoint_config,
        )     # 在eagle3的训练里，self.engine是MegatronEngineWithLMHead的一个实例

        # build dispatch info
        self._register_dispatch_collect_info(
            mesh_name="train",
            dp_rank=self.engine.get_data_parallel_rank(),
            is_collect=self.engine.is_mp_src_rank_with_outputs(),
        )

        if hasattr(self.model_config, "hf_config"):
            self.flops_counter = FlopsCounter(self.model_config.hf_config)
        else:
            self.flops_counter = None

        self.loss_fn = None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def to(self, device, model=True, optimizer=True, grad=True):
        """Manual control of load/offload"""
        assert device in ["cpu", "device"]

        if device == "device":
            device = get_device_name()

        self.engine.to(device=device, model=model, optimizer=optimizer, grad=grad)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_loss_fn(self, loss_fn):
        self.loss_fn = loss_fn

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def reset(self):
        """
        Reset the model engine to the initial state. If the engine is not initialized,
        we initialize it. Otherwise, reload ckpt and reset states
        """
        self.engine.initialize()         # engine初始化，在eagle3的训练里，self.engine是MegatronEngineWithLMHead的一个实例。会跳到transformer_impl.py的MegatronEngine.initialize()

    def _postprocess_output(self, output, *, global_token_num, delta_time, forward_only, images_seqlens):
        """

        Args:
            output: a dictionary containing loss, model_outputs and metrics

        Returns:

        """

        metrics: dict = output.pop("metrics")
        # perform all gather in dp group to ensure that it's correct.
        # Here each metric in metrics can be a list (micro-batch metrics) or a singleton
        # we should always sum the loss of each micro-batch as we scale by global_bsz/global_token
        loss = torch.sum(torch.tensor(output.pop("loss"), device=self.device_name))
        dp_group = self.engine.get_data_parallel_group()
        if dp_group is not None:
            torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG, group=dp_group)
        loss = loss.item()

        # For grad_norm, we do not perform all reduce because it is already been done when clipping grad
        grad_norm = metrics.pop("grad_norm", None)
        if isinstance(grad_norm, torch.Tensor):
            grad_norm = grad_norm.detach().item()
        lr = metrics.pop("lr", None)

        # For other metrics, we perform all gather in dp group (only if DP > 1)
        if dp_group is not None:
            final_metrics = allgather_dict_into_dict(data=metrics, group=dp_group)
        else:
            final_metrics = metrics
        final_metrics["loss"] = loss
        if grad_norm is not None:
            final_metrics["grad_norm"] = grad_norm
        if lr is not None:
            final_metrics["lr"] = lr

        # log memory
        final_metrics["perf/max_memory_allocated_gb"] = get_torch_device().max_memory_allocated() / (1024**3)
        final_metrics["perf/max_memory_reserved_gb"] = get_torch_device().max_memory_reserved() / (1024**3)
        final_metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)

        # TODO: confirm the mtp loss IS same across dp
        for k, v in final_metrics.items():
            if k.startswith("mtp_losses"):
                flatten_v = [sublist[0] for sublist in v]  # sublist should be single element
                final_metrics[k] = sum(flatten_v) / len(flatten_v)
        # compute mfu
        if global_token_num is not None and self.flops_counter is not None:
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(
                global_token_num, delta_time, images_seqlens=images_seqlens
            )
            final_metrics["mfu"] = estimated_flops / promised_flops / torch.distributed.get_world_size()
            if forward_only:
                final_metrics["mfu"] /= 3.0
        # model outputs
        model_output = output.pop("model_output", {})
        # We only return final_metrics
        final_output = tu.get_tensordict(tensor_dict=model_output, non_tensor_dict={"metrics": final_metrics})
        return final_output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="train"), blocking=False)
    def train_mini_batch(self, data: TensorDict) -> TensorDict:
        """Split a batch into N mini-batches run for multiple epochs

        Args:
            data:

        Returns:

        """
        maybe_fix_3d_position_ids(data)
        batch_size_per_dp = data.shape[0]
        disable_auto_offload = tu.pop(data, key="disable_auto_offload", default=False)
        mini_batch_size = tu.pop(data, key="mini_batch_size", default=None)
        num_mini_batch = tu.pop(data, key="num_mini_batch", default=None)
        epochs = tu.pop(data, key="epochs", default=1)
        seed = tu.pop(data, key="seed", default=42)
        dataloader_kwargs = tu.pop(data, key="dataloader_kwargs", default={})

        assert mini_batch_size is not None or num_mini_batch is not None

        if mini_batch_size is None:
            assert batch_size_per_dp % num_mini_batch == 0, f"Got {batch_size_per_dp=} and {num_mini_batch=}"
            mini_batch_size_per_gpu = batch_size_per_dp // num_mini_batch
        else:
            assert mini_batch_size % self.engine.get_data_parallel_size() == 0, (
                f"Got {mini_batch_size=} and {self.engine.get_data_parallel_size()=}"
            )
            mini_batch_size_per_gpu = mini_batch_size // self.engine.get_data_parallel_size()

        # make iterator
        dataloader = tu.make_iterator(
            data,
            mini_batch_size=mini_batch_size_per_gpu,
            epochs=epochs,
            seed=seed + self.engine.get_data_parallel_rank(),
            dataloader_kwargs=dataloader_kwargs,
        )

        with (
            self.engine.train_mode(disable_auto_offload=disable_auto_offload),
            Timer(name="train_batch", logger=None),
        ):
            # update
            output_lst = []
            total_num_iterations = data.shape[0] // mini_batch_size_per_gpu * epochs

            for batch_idx, mini_batch_td in enumerate(dataloader):
                maybe_fix_3d_position_ids(mini_batch_td)
                # add global token num
                if "input_ids" in mini_batch_td:
                    global_token_num = mini_batch_td["input_ids"].offsets().diff().tolist()  # (total_nnz,)
                    # allgather from dp rank
                    global_token_num_output = [None] * torch.distributed.get_world_size(
                        self.engine.get_data_parallel_group()
                    )
                    torch.distributed.all_gather_object(
                        global_token_num_output, global_token_num, self.engine.get_data_parallel_group()
                    )
                    global_token_num = [x for xs in global_token_num_output for x in xs]
                else:
                    global_token_num = None

                tu.assign_non_tensor(
                    mini_batch_td,
                    global_token_num=NonTensorData(global_token_num),
                    update_lr_scheduler=batch_idx == total_num_iterations - 1,
                    disable_auto_offload=True,
                )
                actor_output = self.train_batch(mini_batch_td)
                output_lst.append(actor_output)
                # Advance the profiler schedule once per mini-batch. No-op unless a
                # torch profiler schedule (wait/warmup/active/repeat) is active.
                self.profiler.step()

            if self.engine.is_mp_src_rank_with_outputs():
                actor_output = [tu.get(output, "metrics") for output in output_lst]
                metrics = {}
                for output in actor_output:
                    for key, val in output.items():
                        # flattn dp and micro batch
                        if isinstance(val, list):
                            output[key] = (
                                Metric.aggregate_dp(val)
                                if isinstance(val[0], Metric)
                                else list(chain.from_iterable(val))
                            )
                    append_to_dict(metrics, output)

                output = tu.get_tensordict(tensor_dict={}, non_tensor_dict={"metrics": metrics}).cpu()
            else:
                output = None
        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="train"), blocking=False)
    @DistProfiler.annotate(color="red", role="train_batch")
    def train_batch(self, data: TensorDict) -> TensorDict:
        """训练一个 batch（路由入口）

        v3 下只剩一条路径：原有逻辑 _train_batch_original（并行模式或串行的 actor 步）。
        v1/v2 的 train_draft_only 分支已停用——驱动侧 trainer_base.py:817 恒设该标志为
        False，draft 训练改由 update_draft_deferred 在 update_actor 之后单独驱动。
        """
        # [P3-DEAD v1/v2 20260829] train_draft_only 分支不可达，函数本体已注释。
        # 整体验证通过后连同 _train_batch_draft_only 一并删除。
        # train_draft_only = tu.get_non_tensor_data(data, "train_draft_only", default=False)
        #
        # if train_draft_only:
        #     # 串行模式：Draft 训练步（新增逻辑）
        #     return self._train_batch_draft_only(data)
        # else:
        #     # 原有逻辑（并行模式或串行的 actor 步）
        #     return self._train_batch_original(data)
        return self._train_batch_original(data)

    def _train_batch_original(self, data: TensorDict) -> TensorDict:
        """原有的 train_batch 逻辑（完全不改动，只是重命名）

        【原有逻辑封装】这是原有 train_batch 方法的完整复制。
        所有逻辑完全不变，只是移动到这个独立方法中。

        关闭串行开关后，执行路径会进入此方法，代码 100% 是原有逻辑。
        """
        assert self.loss_fn is not None, "loss function can't be None when calling train_batch"
        assert not self.engine_config.forward_only, "Can't run `train_batch` when forward_only is in the engine config."
        # global_token_num should be a list of number of tokens of each seq in this batch
        global_token_num = tu.get(data, key="global_token_num")
        disable_auto_offload = tu.get(data, key="disable_auto_offload", default=False)
        images_seqlens = tu.get(data, key="images_seqlens", default=None)

        # inject engineering parameters if not specified
        default_keys = dict(
            use_remove_padding=self.model_config.get("use_remove_padding", False),
            use_dynamic_bsz=self.engine_config.use_dynamic_bsz,
            max_token_len_per_gpu=self.engine_config.max_token_len_per_gpu,
            micro_batch_size_per_gpu=self.engine_config.micro_batch_size_per_gpu,
            use_fused_kernels=self.engine_config.use_fused_kernels,
        )

        for key, val in default_keys.items():
            if key not in data.keys():
                tu.assign_non_tensor(data, **{key: val})

        with (
            self.engine.train_mode(disable_auto_offload=disable_auto_offload),
            Timer(name="train_batch", logger=None) as timer,
        ):
            output = self.engine.train_batch(data, loss_function=self.loss_fn)
            # containing loss, model_output and metrics
            # for training, we only care about loss and metrics
        delta_time = timer.last

        update_lr_scheduler = tu.get(data, key="update_lr_scheduler", default=False)
        # update lr scheduler
        if update_lr_scheduler:
            lr = self.engine.lr_scheduler_step()
        else:
            lr = None

        if self.engine.is_mp_src_rank_with_outputs():
            # we don't need model_output in training. Maybe we change out mind later
            output.pop("model_output")
            if lr is not None:
                output["metrics"]["lr"] = lr
            final_output = self._postprocess_output(
                output,
                global_token_num=global_token_num,
                delta_time=delta_time,
                forward_only=False,
                images_seqlens=images_seqlens,
            ).cpu()
        else:
            final_output = None

        return final_output

    # [P3-DEAD v1/v2 20260829] v1/v2 独立 Draft 步的 worker 侧实现。train_batch 的 train_draft_only
    # 分支已停用，无调用点。整体验证通过后删除。
#     def _train_batch_draft_only(self, data: TensorDict) -> TensorDict:
#         """串行模式的 Draft 训练（新增方法）
#
#         【新增逻辑】专门用于 draft 训练步，与原有逻辑完全独立。
#
#         执行流程：
#         1. Actor forward（冻结参数，生成 teacher）
#         2. Draft forward + loss 计算
#         3. Draft backward + 参数更新
#         """
#         assert self.loss_fn is not None, "loss function can't be None when calling train_batch"
#         assert not self.engine_config.forward_only, "Can't run `train_batch` when forward_only is in the engine config."
#
#         # 获取必要参数
#         global_token_num = tu.get(data, key="global_token_num")
#         disable_auto_offload = tu.get(data, key="disable_auto_offload", default=False)
#         images_seqlens = tu.get(data, key="images_seqlens", default=None)
#
#         # 注入工程参数
#         default_keys = dict(
#             use_remove_padding=self.model_config.get("use_remove_padding", False),
#             use_dynamic_bsz=self.engine_config.use_dynamic_bsz,
#             max_token_len_per_gpu=self.engine_config.max_token_len_per_gpu,
#             micro_batch_size_per_gpu=self.engine_config.micro_batch_size_per_gpu,
#             use_fused_kernels=self.engine_config.use_fused_kernels,
#         )
#
#         for key, val in default_keys.items():
#             if key not in data.keys():
#                 tu.assign_non_tensor(data, **{key: val})
#
#         # 设置标志（传递给 engine）
#         tu.assign_non_tensor_data(data, "train_draft_only", True)
#         tu.assign_non_tensor_data(data, "enable_draft_training", True)
#
#         # 训练流程
#         #
#         # 【为什么不直接调 engine.train_batch】
#         # engine/base.py 的 train_batch 是 zero_grad -> forward_backward(forward_only=False)
#         # -> optimizer_step() 三件套，对串行 Draft 步有两处不适用：
#         #
#         #   1) optimizer_step() 是 **policy** 的 optimizer，没有任何 train_draft_only 守卫。
#         #      串行 Draft 步的语义是"policy 只做 forward 产 teacher，不更新"，直接调
#         #      train_batch 会把 policy 一起用 GRPO 更新掉，与串行设计相悖。
#         #   2) forward_only=False 会让 megatron 对 loss_function 的返回值做 backward，
#         #      而 loss_fn 是 ppo_loss，它需要 old_log_probs / advantages
#         #      （losses.py: data.select("response_mask", "old_log_probs", "advantages")）。
#         #      串行 Draft 分支只跑了 reward + _balance_batch，从没调过 _compute_old_log_prob
#         #      和 _compute_advantage，这两个字段根本不存在 -> KeyError。
#         #
#         # 所以这里改用 forward_only=True 手工展开：
#         #   - loss_function=None + forward_only=True -> postprocess 走常量 loss 分支，
#         #     ppo_loss 永不被调用，问题 2 消失；
#         #   - megatron 在 forward_only 下只是跳过 backward_step，**不加 no_grad**
#         #     （schedules.py:634-652 已确认），所以 draft 的计算图完好存活；
#         #   - policy 全程不 backward，问题 1 消失，且省掉一整个反向的算力/显存；
#         #   - draft 的 backward + draft_optimizer.step() 由 eagle3_backward_step 负责，
#         #     它在 forward_backward_batch 内部执行（该处守卫已放宽为
#         #     `not forward_only or train_draft_only`，以接纳本路径）。串行与并行共用
#         #     同一个调用点和同一套 metrics 包装，本方法不再手工补调。
#         with (
#             self.engine.train_mode(disable_auto_offload=disable_auto_offload),
#             Timer(name="train_batch_draft", logger=None) as timer,
#         ):
#             maybe_fix_3d_position_ids(data)
#             # draft 的 backward + draft_optimizer.step()，以及 draft_loss 写入 metrics，
#             # 均由 forward_backward_batch 内部的 eagle3_backward_step 完成
#             # （transformer_impl.py:892，该处守卫已放宽以接纳 forward_only=True 的串行 Draft 步）。
#             # 注入点在 :936 postprocess_batch_func 之前，metrics 由 append_to_dict 统一包成
#             # list —— 与并行完全同一条路径，串行不再自持形状责任。
#             #
#             # 【不要在这里手工补调 eagle3_backward_step】那样注入点会落到
#             # postprocess_batch_func 之后，跳过 append_to_dict 的包装，裸 float 进
#             # allgather_dict_into_dict 后会被 train_mini_batch:326 的 chain.from_iterable
#             # 炸掉（'float' object is not iterable）。2026-08-28 已经这么错过一次，详见 优化16。
#             output = self.engine.forward_backward_batch(data, loss_function=None, forward_only=True)
#
#         delta_time = timer.last
#
#         # 处理输出
#         if self.engine.is_mp_src_rank_with_outputs():
#             output.pop("model_output", None)
#
#             # 【守卫】串行 Draft 步的唯一目的就是训 draft。eagle3_backward_step() 在
#             # 取不到暂存 loss 时会 return None 并静默跳过 backward/step，本步就会整步
#             # 空转却上报"成功"，安静烧掉一轮训练时间（2026-08-26 就这么浪费过一轮）。
#             # 这里显式检查 draft_loss 是否真的产生，没有就直接失败，不允许静默。
#             if "draft_loss" not in output.get("metrics", {}):
#                 raise RuntimeError(
#                     "[DRAFT-TRAIN-SERIAL] Draft 训练步没有产生 draft_loss：本步 draft 未做 "
#                     "backward/optimizer.step()，属于整步空转。请检查 eagle3_patch 的 draft "
#                     "前向是否被跳过（如 loss_mask is None），或 _eagle3_draft_losses 是否被 "
#                     "其他代码提前 drain。"
#                 )
#
#             final_output = self._postprocess_output(
#                 output,
#                 global_token_num=global_token_num,
#                 delta_time=delta_time,
#                 forward_only=False,
#                 images_seqlens=images_seqlens,
#             ).cpu()
#         else:
#             final_output = None
#
#         return final_output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="train"), blocking=False)
    def infer_batch(self, data: TensorDict) -> TensorDict:
        # add mfu calculator
        global_token_num = tu.get(data, key="global_token_num")
        compute_loss = tu.get(data, key="compute_loss", default=True)
        disable_auto_offload = tu.get(data, key="disable_auto_offload", default=False)
        no_lora_adapter = tu.pop(data, key="no_lora_adapter", default=False)
        images_seqlens = tu.get(data, key="images_seqlens", default=None)

        default_keys = dict(
            use_remove_padding=self.model_config.get("use_remove_padding", False),
            use_dynamic_bsz=self.engine_config.use_dynamic_bsz,
            max_token_len_per_gpu=self.engine_config.infer_max_token_len_per_gpu,
            micro_batch_size_per_gpu=self.engine_config.infer_micro_batch_size_per_gpu,
            use_fused_kernels=self.engine_config.use_fused_kernels,
        )

        for key, val in default_keys.items():
            if key not in data.keys():
                tu.assign_non_tensor(data, **{key: val})

        # for sft training, we need to compute loss in eval
        loss_function = self.loss_fn if compute_loss else None

        with (
            self.engine.eval_mode(disable_auto_offload=disable_auto_offload),
            Timer(name="eval_batch", logger=None) as timer,
        ):
            adapter_ctx = self.engine.disable_adapter() if no_lora_adapter else nullcontext()
            with adapter_ctx:
                output = self.engine.infer_batch(data, loss_function=loss_function)
        delta_time = timer.last

        if self.engine.is_mp_src_rank_with_outputs():
            final_output = self._postprocess_output(
                output,
                global_token_num=global_token_num,
                delta_time=delta_time,
                forward_only=True,
                images_seqlens=images_seqlens,
            ).cpu()
        else:
            final_output = None

        return final_output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        return self.engine.save_checkpoint(local_path, hdfs_path, global_step, max_ckpt_to_keep)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        return self.engine.load_checkpoint(local_path, hdfs_path, del_local_after_load)


class ActorRolloutRefWorker(Worker, DistProfilerExtension):
    """Hybrid worker that includes actor model, rollout and optional ref model.
    For standalone actor or rollout, use ActorWorker or BaseRollout respectively.

    NOTE: ActorRolloutRefWorker no longer support spmd mode and run native server mode.
    """

    actor_worker_cls = TrainingWorker
    ref_worker_cls = TrainingWorker

    def __init__(
        self, config: DictConfig, role: str, distillation_config: Optional[DistillationConfig] = None, **kwargs
    ):
        Worker.__init__(self)
        self.config = config
        self.distillation_config = distillation_config
        self.distillation_enabled = is_distillation_enabled(distillation_config)
        self.role = role
        self.actor: TrainingWorker | None = None
        self.ref: TrainingWorker | None = None
        self.rollout: BaseRollout = None
        assert self.role in ["actor", "rollout", "ref", "actor_rollout", "actor_rollout_ref"]
        self._is_actor = self.role in ["actor", "actor_rollout", "actor_rollout_ref"]
        self._is_rollout = self.role in ["rollout", "actor_rollout", "actor_rollout_ref"]
        self._is_ref = self.role in ["ref", "actor_rollout_ref"]

        if self._is_actor:
            omega_profiler_config = config.actor.get("profiler", {})
        elif self._is_rollout:
            # NOTE: In colocation mode, rollout config may not take effect (follow the actor config)
            # This is for extendability in AsyncRL cases
            omega_profiler_config = config.rollout.get("profiler", {})
        else:
            omega_profiler_config = config.ref.get("profiler", {})

        profiler_config = omega_conf_to_dataclass(omega_profiler_config, dataclass_type=ProfilerConfig)
        if omega_profiler_config.get("tool", None) in ["npu", "nsys", "torch", "torch_memory", "precision_debugger"]:
            tool_config = omega_conf_to_dataclass(
                omega_profiler_config.get("tool_config", {}).get(omega_profiler_config.get("tool"))
            )
        else:
            tool_config = None

        # Router replay is supported on the megatron engine and on the veomni
        # engine. Both expose `router_replay` on their per-strategy engine
        # config (the field lives on the shared `EngineConfig` base).
        actor_strategy = self.config.actor.strategy
        if actor_strategy == "megatron":
            rr_mode = self.config.actor.megatron.router_replay.mode
        elif actor_strategy == "veomni":
            rr_mode = self.config.actor.veomni.router_replay.mode
        else:
            rr_mode = "disabled"
        self.enable_routing_replay = rr_mode != "disabled"

        # Keep the raw (un-dataclassed) role profiler config so the inner actor
        # TrainingWorker can build a matching DistProfiler in init_model. This lets
        # train_mini_batch drive the (process-global) torch profiler schedule via
        # profiler.step(), even though start/stop happen on this outer worker.
        # NOTE: we must rebuild via the hydra path (omega_conf_to_dataclass without
        # dataclass_type) so that tool_config entries are real dataclasses with
        # attribute access; the dataclass_type=ProfilerConfig variant above yields a
        # plain-dict tool_config that the inner torch profiler cannot consume.
        self._omega_profiler_config = omega_profiler_config

        DistProfilerExtension.__init__(
            self,
            DistProfiler(
                rank=self.rank,
                config=profiler_config,
                tool_config=tool_config,
                # Embed the worker role (actor/rollout/ref/...) in trace filenames so
                # per-process results are distinguishable across roles and ranks.
                save_file_prefix=self.role,
            ),
        )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_loss_fn(self, loss_fn):
        self.actor.set_loss_fn(loss_fn=loss_fn)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def to(self, device, model=True, optimizer=True, grad=True):
        """Manual control of load/offload"""
        self.actor.to(device=device, model=model, optimizer=optimizer, grad=grad)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        model_config: HFModelConfig = omega_conf_to_dataclass(self.config.model)

        # 1. build reference model
        if "ref" in self.role:
            # TODO: align ref config with actor config
            with open_dict(self.config.ref):
                self.config.ref.ppo_mini_batch_size = self.config.actor.ppo_mini_batch_size
                self.config.ref.ppo_micro_batch_size = self.config.ref.pop("log_prob_micro_batch_size", None)
                self.config.ref.ppo_micro_batch_size_per_gpu = self.config.ref.pop(
                    "log_prob_micro_batch_size_per_gpu", None
                )
                self.config.ref.use_dynamic_bsz = self.config.ref.pop("log_prob_use_dynamic_bsz", False)
                self.config.ref.ppo_max_token_len_per_gpu = self.config.ref.pop("log_prob_max_token_len_per_gpu", None)
            ref_config: ActorConfig = omega_conf_to_dataclass(self.config.ref)

            # The ref model does not need to enable MTP; force it to false.
            ref_config.model_config = deepcopy(model_config)
            ref_config.model_config.mtp = MtpConfig(enable=False)

            # Build the inner ref profiler config via the hydra path (same as the actor / SFT),
            # so its tool_config entries are real dataclass instances the torch profiler can read.
            # This puts the reference model's inner TrainingWorker on par with the actor's, so the
            # torch profiler (and the nsys/npu backends) support the reference model too, instead
            # of the ref silently running with a disabled no-op profiler.
            ref_omega_profiler_config = self.config.ref.get("profiler", {})
            ref_profiler_config = (
                omega_conf_to_dataclass(ref_omega_profiler_config) if ref_omega_profiler_config else None
            )

            # construct TrainingWorkerConfig
            ref_training_config = TrainingWorkerConfig(
                model_type=ref_config.model_config.get("model_type", "language_model"),
                model_config=ref_config.model_config,
                engine_config=ref_config.engine,
                optimizer_config=ref_config.optim,
                checkpoint_config=ref_config.checkpoint,
                profiler_config=ref_profiler_config,
            )

            # assign engine configs
            ref_training_config.engine_config.use_dynamic_bsz = self.config.ref.use_dynamic_bsz
            ref_training_config.engine_config.infer_max_token_len_per_gpu = self.config.ref.ppo_max_token_len_per_gpu
            ref_training_config.engine_config.infer_micro_batch_size_per_gpu = (
                self.config.ref.ppo_micro_batch_size_per_gpu
            )
            ref_training_config.engine_config.use_remove_padding = model_config.get("use_remove_padding", False)

            self.ref = self.ref_worker_cls(config=ref_training_config)
            self.ref.reset()
            self.set_dispatch_collect(mesh_name="ref", **self.ref.get_dispatch_collect())

        # 2. build actor model
        if "actor" in self.role:
            actor_config: ActorConfig = omega_conf_to_dataclass(self.config.actor)
            actor_config.model_config = model_config

            # === 串行训练：存到 self 供 v3 的 draft 采集/训练配置读取（见 _eagle3_collect_config、
            #     update_draft_deferred）。原注释说的 _apply_draft_batch_config 已是 v1/v2 死代码。===
            self.actor_config = actor_config

            distillation_config: Optional[DistillationConfig] = (
                omega_conf_to_dataclass(self.distillation_config) if self.distillation_enabled else None
            )

            # Build the inner actor profiler config via the hydra path (same as SFT), so
            # its tool_config entries are real dataclass instances the torch profiler can
            # read. This gives the inner TrainingWorker a DistProfiler that shares the
            # process-global torch profiler, so per-mini-batch profiler.step() works.
            actor_profiler_config = (
                omega_conf_to_dataclass(self._omega_profiler_config) if self._omega_profiler_config else None
            )

            actor_training_config = TrainingWorkerConfig(
                model_type=actor_config.model_config.get("model_type", "language_model"),
                model_config=actor_config.model_config,
                engine_config=actor_config.engine,
                optimizer_config=actor_config.optim,
                checkpoint_config=actor_config.checkpoint,
                profiler_config=actor_profiler_config,
            )

            assert self.config.actor.use_dynamic_bsz == self.config.rollout.log_prob_use_dynamic_bsz

            # assign engine configs
            actor_training_config.engine_config.use_dynamic_bsz = self.config.actor.use_dynamic_bsz
            actor_training_config.engine_config.infer_max_token_len_per_gpu = (
                self.config.rollout.log_prob_max_token_len_per_gpu
            )
            actor_training_config.engine_config.infer_micro_batch_size_per_gpu = (
                self.config.rollout.log_prob_micro_batch_size_per_gpu
            )
            actor_training_config.engine_config.max_token_len_per_gpu = self.config.actor.ppo_max_token_len_per_gpu
            actor_training_config.engine_config.micro_batch_size_per_gpu = (
                self.config.actor.ppo_micro_batch_size_per_gpu
            )
            actor_training_config.engine_config.use_remove_padding = model_config.get("use_remove_padding", False)

            if self.config.actor.use_dynamic_bsz:
                assert self.config.rollout.log_prob_max_token_len_per_gpu is not None
                assert self.config.actor.ppo_max_token_len_per_gpu is not None
            else:
                assert self.config.rollout.log_prob_micro_batch_size_per_gpu is not None
                assert self.config.actor.ppo_micro_batch_size_per_gpu is not None
            if self.distillation_enabled:
                self.loss_fn = partial(
                    distillation_ppo_loss, config=actor_config, distillation_config=distillation_config
                )
            else:
                self.loss_fn = partial(ppo_loss, config=actor_config)
            self.actor = self.actor_worker_cls(config=actor_training_config)    # 建内层 actor worker，actor_worker_cls = TrainingWorker
            self.actor.reset()                                                  # ← 往这，RPC 到内层 worker，跳到TrainingWorker.reset()
            self.actor.set_loss_fn(self.loss_fn)
            self.set_dispatch_collect(mesh_name="actor", **self.actor.get_dispatch_collect())

        # 3. build rollout engine
        if "rollout" in self.role:
            rollout_config: RolloutConfig = omega_conf_to_dataclass(self.config.rollout)

            # TODO: move rollout_device_mesh into ServerAdapter
            # 3.1 build rollout device mesh (sglang need only)
            infer_tp = rollout_config.tensor_model_parallel_size * rollout_config.data_parallel_size
            infer_pp = rollout_config.pipeline_model_parallel_size
            infer_world_size = infer_tp * infer_pp
            dp = self.world_size // infer_world_size
            assert self.world_size % infer_world_size == 0, (
                f"rollout world_size: {self.world_size} is not divisible by infer_world_size: {infer_world_size}"
            )
            rollout_device_mesh = init_device_mesh(
                get_device_name(), mesh_shape=(dp, infer_tp, infer_pp), mesh_dim_names=["dp", "infer_tp", "infer_pp"]
            )

            # 3.2 initialize rollout engine
            rollout_cls: type[BaseRollout] = get_rollout_class(rollout_config.name, rollout_config.mode)
            self.rollout = rollout_cls(
                config=rollout_config, model_config=model_config, device_mesh=rollout_device_mesh
            )

            # used for LoRA (base_sync_done is unused in merge-only mode but kept for Phase 2 adapter path)
            self.base_sync_done: bool = "dummy" not in self.config.rollout.load_format
            self.layered_summon = self.config.rollout.get("layered_summon", False)
            self.peft_merge: bool = model_config.lora.get("merge", False)

        # 4. build checkpoint engine
        if "actor" in self.role:
            checkpoint_engine_config = omega_conf_to_dataclass(self.config.rollout.checkpoint_engine)
            backend = checkpoint_engine_config.backend
            bucket_size = checkpoint_engine_config.update_weights_bucket_megabytes << 20
            engine_kwargs = checkpoint_engine_config.engine_kwargs.get(backend, {})
            # If custom_backend_module is set, import it so plugins can register
            # in CheckpointEngineRegistry before the backend is instantiated.
            import_external_libs(checkpoint_engine_config.custom_backend_module or None)
            self.checkpoint_engine = CheckpointEngineRegistry.new(
                backend, is_master=(torch.distributed.get_rank() == 0), bucket_size=bucket_size, **engine_kwargs
            )

        # Free cached GPU memory so colocated vLLM processes can see it via cudaMemGetInfo
        aggressive_empty_cache(force_sync=True)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="ref"))
    @DistProfiler.annotate(color="olive", role="ref_compute_log_prob")
    @_with_routing_replay_flag(enabled=False)
    def compute_ref_log_prob(self, data: TensorDict) -> TensorDict:
        output = self.ref.infer_batch(data=data)
        return output.cpu() if output is not None else None

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="blue", role="actor_compute_log_prob")
    @_with_routing_replay_flag(enabled=True)
    def compute_log_prob(self, data: TensorDict) -> TensorDict:
        # v3 deferred draft training: this forward is the one PPO already pays
        # for, so it is where draft features get harvested. Opening the store
        # before the forward and leaving it filled afterwards is the whole
        # handoff -- update_draft_deferred drains it later in the same step.
        self._eagle3_open_collection(data)
        output = self.actor.infer_batch(data)

        return output.cpu() if output is not None else None

    def _eagle3_open_collection(self, data: TensorDict) -> bool:
        """Arm feature collection for the upcoming log-prob forward.

        Returns False (and leaves the flag off) whenever deferred training is not
        active for this step, so the forward takes the ordinary path.
        """
        engine = getattr(self.actor, "engine", None)
        state = getattr(engine, "_eagle3", None)
        if state is None or not state.enabled:
            return False
        if not tu.get_non_tensor_data(data, "eagle3_collect_only", default=False):
            return False

        from verl.models.eagle3.feature_store import DraftFeatureStore

        global_step = _eagle3_scalar_global_step(data)
        if state.feature_store is None:
            state.feature_store = DraftFeatureStore()
        # begin_step drops anything a previous step left behind and warns if it
        # was never drained -- features harvested for nothing.
        state.feature_store.begin_step(global_step)
        cfg = self._eagle3_collect_config(global_step)
        state.collect_config = cfg

        # Reset the step's quota here, NOT in _postprocess: that runs once per
        # micro-batch, and at micro_batch_size_per_gpu=1 each call sees a single
        # sequence, so a per-call quota of 16 can never bind. Resetting per
        # micro-batch is what let a step collect one window per sequence
        # (64 per rank) instead of the 16 budgeted.
        from verl.models.eagle3.collect_plan import CollectBudget

        state.collect_budget = CollectBudget(
            max_samples=cfg["max_samples_per_replica"],
            max_tokens=cfg["max_tokens_per_replica"],
        )
        return True

    def _eagle3_collect_config(self, global_step: int) -> dict:
        """Collect-plan knobs, defaulting to the verl-SpeCo values.

        Mirrors verl_speco/config/speco_base.yaml:58-66 and exposes them via
        ActorConfig.draft_collect_* fields so callers can override via hydra
        without touching code (P2#4 panel).
        """
        cfg = self.actor_config
        _g = lambda attr, default: (  # noqa: E731
            v if (v := getattr(cfg, attr, None)) is not None else default
        )
        return {
            "global_step": global_step,
            "window_train_rows": int(_g("draft_collect_window_train_rows", 512)),
            "window_mode": str(_g("draft_collect_window_mode", "front")),
            "sample_rate": float(_g("draft_collect_sample_rate", 1.0)),
            "max_samples_per_replica": int(_g("draft_collect_max_samples_per_replica", 16)),
            "max_tokens_per_replica": int(_g("draft_collect_max_tokens_per_replica", 16384)),
        }

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="orange", role="draft_deferred")
    def update_draft_deferred(self, data: TensorDict) -> TensorDict:
        """Train the draft on features stashed during this step's log-prob forward.

        Call after update_actor has returned. By then the policy's activations and
        gradients are released, so the draft trains beside the policy peak rather
        than on top of it -- the memory property the old alternating-step design
        bought at the cost of a whole extra rollout.

        The teacher snapshot is NOT taken here -- see
        :meth:`snapshot_draft_teacher`, which must run before update_actor.
        """
        from verl.models.eagle3.deferred_training import train_draft_from_store

        engine = self.actor.engine
        state = getattr(engine, "_eagle3", None)
        if state is None or not state.enabled or state.feature_store is None:
            return None

        global_step = _eagle3_scalar_global_step(data)
        micro_bsz = getattr(self.actor_config, "draft_ppo_micro_batch_size_per_gpu", None)
        steps_per_trigger = int(getattr(self.actor_config, "draft_steps_per_trigger", None) or 10)
        train_bsz = getattr(self.actor_config, "draft_train_batch_size_per_gpu", None)
        train_bsz = int(train_bsz) if train_bsz else 4

        # 把 global_step 挂到 engine 上，供 eagle3_backward_step 记录 last_trained_global_step
        # (#2 权重只在训过的步同步，镜像 SpeCo speco_worker.py:921 的 last_trained_step）
        engine._eagle3_last_global_step = global_step

        with engine.train_mode(disable_auto_offload=True), Timer(name="draft_deferred", logger=None) as timer:
            result = train_draft_from_store(
                engine,
                state.feature_store,
                micro_batch_size=micro_bsz,
                global_step=global_step,
                steps_per_trigger=steps_per_trigger,
                batch_size_per_gpu=train_bsz,
            )

        if result is None or not engine.is_mp_src_rank_with_outputs():
            return None
        losses = result["losses"]
        metrics = {
            # draft_loss 取触发内均值；first/last 用于观察单次触发内是否真的在学
            "draft_loss": [sum(losses) / len(losses)],
            "draft_loss_first": [losses[0]],
            "draft_loss_last": [losses[-1]],
            "draft_updates": [float(len(losses))],
            "draft_windows": [float(result["num_windows"])],
            "draft_time_s": [timer.last],
        }
        return tu.get_tensordict(tensor_dict={}, non_tensor_dict={"metrics": metrics}).cpu()

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def snapshot_draft_teacher(self, data: TensorDict) -> None:
        """Snapshot the policy lm_head **before** update_actor runs.

        Ordering here is load-bearing and the failure is silent.

        The hidden states this step stashed were produced during compute_log_prob,
        i.e. by the weights as they stood *before* any of this step's mini-batch
        updates. The teacher is rebuilt as ``lm_head @ stashed_hidden``, so the
        head has to come from that same set of weights. Snapshotting after
        update_actor pairs a post-update head with a pre-update body -- a
        combination that never existed as a model, so the distribution the draft
        is distilled toward is not any policy's.

        Nothing raises if this is skipped or ordered wrongly. The draft trains
        against a slightly wrong target and the only symptom is an acceptance
        rate that does not climb. verl-SpeCo pins the same ordering at
        speco_ray_trainer.py:1891 (sync) vs :1895 (update).
        """
        from verl.models.eagle3.deferred_training import refresh_frozen_teacher_head

        engine = getattr(self.actor, "engine", None)
        state = getattr(engine, "_eagle3", None)
        if state is None or not state.enabled:
            return None
        global_step = _eagle3_scalar_global_step(data)
        # 必须在 engine 的 mode 上下文内跑。param_offload=True 时，policy 参数只在
        # train_mode/eval_mode 内被搬到设备上（BaseEngineCtx._context_switch），
        # 上下文之外 output_layer.weight 是 CPU 张量；而 refresh_frozen_teacher_head
        # 要对它做 TP all_gather，拿 CPU 张量进 HCCL 会直接失败——真机表现为
        # frozen_teacher.py 的 HcclAllgather RuntimeError（异步算子，报错点还会被延后
        # 到下一行的索引，很难从栈上看出真因）。TP=1 时不做 all_gather，所以只有
        # TP>1 才暴露，单元测试也照不到。
        #
        # 用 eval_mode 而不是 train_mode：这里只**读** lm_head，不训练。eval 路径
        # 只 load 模型参数（load_grad=False），跳过优化器状态——对 8B 模型来说
        # Adam state 是权重的两倍，白搬一趟。也不会像 train_mode 那样在退出时
        # zero_grad。
        with engine.eval_mode():
            refresh_frozen_teacher_head(engine, global_step=global_step)
        return None

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="red", role="actor_update")
    @_with_routing_replay_flag(enabled=True)
    def update_actor(self, data: TensorDict) -> TensorDict:
        output = self.actor.train_mini_batch(data=data)
        return output.cpu() if output is not None else None

    # [P3-DEAD v1/v2 20260829] update_draft / _apply_draft_batch_config：v1/v2 独立 Draft 步入口，
    # 唯一调用者 trainer_base._update_draft 已一并停用。整体验证通过后删除。
#     @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
#     @DistProfiler.annotate(color="orange", role="draft_update")
#     @_with_routing_replay_flag(enabled=True)
#     def update_draft(self, data: TensorDict) -> TensorDict:
#         """Draft 专属训练入口（新增方法）
#
#         与 update_actor 并列，专门用于 draft 训练步。
#         本方法是新增的，不修改任何原有方法。
#
#         Args:
#             data: 包含训练数据的 TensorDict
#
#         Returns:
#             训练结果（metrics 等）
#         """
#         # === 新增：应用 Draft 专用的 batch size 配置 ===
#         data = self._apply_draft_batch_config(data)
#
#         # 设置标志：告诉 engine 这是 draft 训练模式
#         tu.assign_non_tensor_data(data, "train_draft_only", True)
#         tu.assign_non_tensor_data(data, "enable_draft_training", True)
#
#         # 调用 actor 的 train_mini_batch（内部会根据标志走 draft 路径）
#         output = self.actor.train_mini_batch(data=data)
#         return output.cpu() if output is not None else None
#
#     def _apply_draft_batch_config(self, data: TensorDict) -> TensorDict:
#         """应用 Draft 专用的 batch size 配置
#
#         从配置中读取 draft_ppo_* 参数，如果存在则覆盖 data 中的对应参数。
#         如果 draft_ppo_* 参数为 None，则使用 Actor 的参数（fallback 机制）。
#
#         注意：train_mini_batch() 期望的参数名是 mini_batch_size / num_mini_batch，
#         不是 ppo_mini_batch_size。
#
#         Args:
#             data: 输入的 TensorDict
#
#         Returns:
#             应用 Draft 配置后的 TensorDict
#         """
#         config = self.actor_config  # ActorConfig 对象
#
#         # 读取 Draft 专用配置（如果存在）
#         draft_mini_bsz = getattr(config, 'draft_ppo_mini_batch_size', None)
#         draft_micro_bsz = getattr(config, 'draft_ppo_micro_batch_size', None)
#         draft_micro_bsz_per_gpu = getattr(config, 'draft_ppo_micro_batch_size_per_gpu', None)
#         draft_infer_micro_bsz_per_gpu = getattr(config, 'draft_ppo_infer_micro_batch_size_per_gpu', None)
#
#         # 应用 Draft 配置（如果设置了的话）
#         # train_mini_batch() 从 data 中读取 "mini_batch_size" 或 "num_mini_batch"
#         if draft_mini_bsz is not None:
#             tu.assign_non_tensor_data(data, "mini_batch_size", draft_mini_bsz)
#             logger.info(f"[Draft Config] Using draft_ppo_mini_batch_size={draft_mini_bsz}")
#
#         # train_mini_batch() 没有直接读取 ppo_micro_batch_size，但可能在其他地方需要
#         # 为了兼容性，两个都设置
#         if draft_micro_bsz_per_gpu is not None:
#             tu.assign_non_tensor_data(data, "ppo_micro_batch_size_per_gpu", draft_micro_bsz_per_gpu)
#             logger.info(f"[Draft Config] Using draft_ppo_micro_batch_size_per_gpu={draft_micro_bsz_per_gpu}")
#
#         if draft_infer_micro_bsz_per_gpu is not None:
#             tu.assign_non_tensor_data(data, "ppo_infer_micro_batch_size_per_gpu", draft_infer_micro_bsz_per_gpu)
#             logger.info(f"[Draft Config] Using draft_ppo_infer_micro_batch_size_per_gpu={draft_infer_micro_bsz_per_gpu}")
#
#         return data

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        assert "actor" in self.role, "load_checkpoint only support actor role"
        self.actor.load_checkpoint(local_path, hdfs_path, del_local_after_load)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        assert "actor" in self.role, "save_checkpoint only support actor role"
        self.actor.save_checkpoint(local_path, hdfs_path, global_step, max_ckpt_to_keep)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def update_weights(self, global_steps: int = None, mode: str = "auto"):
        """Update weights from trainer to rollout.

        1. For sync training with colocated trainer and rollout, update rollout directly from model engine.
           - before update_weights: rollout should be in sleep mode.
           - after update_weights: rollout should be in wake_up mode.
        2. For async training with disaggregated trainer and rollout, send_weights only by checkpoint engine.

        LoRA handling: when model.lora.merge=True (peft_merge), LoRA is merged into
        base weights before sync. The engine returns full HF-keyed params with
        peft_config=None, so the rollout receives a standard weight update.

        Args:
            global_steps: Current global training step count, passed to rollout for logging/tracking.
            mode: Weight update strategy. Supported values:
                - ``"auto"``: Automatically resolve to the backend configured in
                  ``config.rollout.checkpoint_engine.backend`` (default).
                - ``"naive"``: Direct in-process weight sync between colocated trainer
                  and rollout. Used for synchronous training where both share the same
                  process. Rollout must be in sleep mode before this call.
                - Any other value: Delegates to
                  :meth:`checkpoint_engine.send_weights` for asynchronous weight
                  transfer via checkpoint engine, suitable for disaggregated
                  trainer/rollout deployments.
        """

        # Resolve mode: "auto" falls back to config, explicit values take precedence
        effective_mode = mode if mode != "auto" else self.config.rollout.checkpoint_engine.backend

        # 0. send_weights only for async training with disaggregated trainer and rollout
        if effective_mode != "naive":
            # The sharded delta engine diffs each rank's local FSDP shard (no all-gather),
            # so it consumes the sharded param generator instead of the full-tensor one.
            if effective_mode == "delta_sharded":
                per_tensor_param, _ = self.actor.engine.get_per_tensor_param_shard()
            else:
                per_tensor_param, _ = self.actor.engine.get_per_tensor_param()
            metrics = await self.checkpoint_engine.send_weights(per_tensor_param, global_steps=global_steps)
            return metrics or {}

        set_expandable_segments(False)
        log_gpu_memory_usage("Before resume weights", logger=logger)

        # 1. resume rollout memory (weights were released during sleep)
        if self.config.rollout.free_cache_engine:
            await self.rollout.resume(tags=["weights"])
        log_gpu_memory_usage("After resume weights", logger=logger)

        # 2. determine if we need a base weight sync (adapter path only)
        per_tensor_param, peft_config = self.actor.engine.get_per_tensor_param(
            layered_summon=self.layered_summon, base_sync_done=True
        )

        do_lora_base_sync = False
        if not self.peft_merge and peft_config is not None:
            self.rollout.sleep_level = 1
            do_lora_base_sync = not self.base_sync_done

        # 3. sync weights: For SGLang, we need base first (when needed), then adapter/merged
        if do_lora_base_sync:
            per_tensor_param_base, peft_config = self.actor.engine.get_per_tensor_param(
                layered_summon=self.layered_summon, base_sync_done=False
            )
            await self.rollout.update_weights(
                per_tensor_param_base, peft_config=peft_config, base_sync_done=False, global_steps=global_steps
            )

        await self.rollout.update_weights(
            per_tensor_param, peft_config=peft_config, base_sync_done=True, global_steps=global_steps
        )

        log_gpu_memory_usage("After update_weights", logger=logger)

        # 3. offload model to cpu
        if self.actor.engine.is_param_offload_enabled:
            self.actor.engine.to("cpu", model=True, optimizer=False, grad=False)
        aggressive_empty_cache(force_sync=True)

        # 4. resume kv_cache
        if self.config.rollout.free_cache_engine:
            await self.rollout.resume(tags=["kv_cache"])
        log_gpu_memory_usage("After resume kv_cache", logger=logger)

        self.base_sync_done = True
        set_expandable_segments(True)

    @register(dispatch_mode=Dispatch.DP_COMPUTE, blocking=False)
    def execute_checkpoint_engine(self, method: str, *args, **kwargs):
        """Execute checkpoint engine method.

        Args:
            method (str): Checkpoint engine method name.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        """
        return getattr(self.checkpoint_engine, method)(*args, **kwargs)
