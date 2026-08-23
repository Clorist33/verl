#!/usr/bin/env bash
# EAGLE3 | Qwen3 (dense) | draft-model online training | Megatron backend | Ascend NPU
# ---------------------------------------------------------------------------
# First-version gate: only PP>1 is blocked (engine_support.py:143). TP/EP/ETP are
# unrestricted. The draft has its own independent optimizer; by default it is a
# replicated nn.Module (untouched by policy tensor/expert parallelism), or -- with
# USE_MEGATRON_DRAFT=True -- a MegatronModule whose backbone is TP-sharded.
#
# TWO model paths are passed SEPARATELY (they are NOT one combined input):
#   actor_rollout_ref.model.path                        -> POLICY (weights required)
#   actor_rollout_ref.model.eagle3.draft_model_path     -> DRAFT  (config required, weights optional)
#
# Usage:
#   POLICY_PATH=/path/to/Qwen3-8B \
#   DRAFT_PATH=/path/to/eagle3_draft_ckpt \
#     bash examples/eagle3/run_qwen3_eagle3_megatron.sh
#
# What this run proves: the draft loss (metric `draft_loss`) goes DOWN while the
# policy trains normally with GRPO. It does NOT turn on rollout-side speculative
# decoding (enable_rollout=False): that needs refit + a matching draft ckpt.
# Flip EAGLE3_ENABLE_ROLLOUT=True once the draft is trained and refit is wired.
# ---------------------------------------------------------------------------
set -xeuo pipefail

# ===== 关键环境变量必须最先设置 =====
# 这些环境变量必须在任何进程启动前设置，确保所有子进程都能继承
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
export HCCL_CONNECT_TIMEOUT=1500
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60500
export HCCL_NPU_SOCKET_PORT_RANGE=61000-62000
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export TASK_QUEUE_ENABLE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export DISABLE_L2_CACHE=1
export HCCL_OP_EXPANSION_MODE=${HCCL_OP_EXPANSION_MODE:-AIV}
export VLLM_USE_V1=1

# ============================================
# Ray 集群预检查和启动（修复 4 GPU 和 HCCL 端口冲突问题）
# ============================================
echo "=========================================="
echo "Pre-flight check: Ensuring Ray cluster with 16 GPUs"
echo "=========================================="

# 检查 Ray 是否已运行且配置正确
RAY_RUNNING=false
if ray status >/dev/null 2>&1; then
    CURRENT_GPUS=$(python3 -c "import ray; ray.init(address='auto'); g=ray.cluster_resources().get('GPU',0); ray.shutdown(); print(g)" 2>/dev/null || echo "0")
    if [ "$CURRENT_GPUS" = "16.0" ]; then
        echo "✓ Ray cluster already running with 16 GPUs"
        RAY_RUNNING=true
    else
        echo "✗ Ray cluster running but only has $CURRENT_GPUS GPUs (expected 16)"
        echo "  Restarting Ray with correct configuration..."
        ray stop --force 2>/dev/null || true
        sleep 3
    fi
else
    echo "✗ Ray cluster not running"
fi

# 启动 Ray（如果需要）
if [ "$RAY_RUNNING" = false ]; then
    echo "Starting Ray cluster with 16 GPUs..."
    ray start --head --num-gpus=16 --resources='{"NPU": 16}'
    sleep 2

    # 验证启动成功
    VERIFY_GPUS=$(python3 -c "import ray; ray.init(address='auto'); g=ray.cluster_resources().get('GPU',0); ray.shutdown(); print(g)" 2>/dev/null || echo "0")
    if [ "$VERIFY_GPUS" = "16.0" ]; then
        echo "✓ Ray cluster successfully started with 16 GPUs"
    else
        echo "✗ ERROR: Ray started but only has $VERIFY_GPUS GPUs"
        echo "  Please manually run: ray stop && ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 ray start --head --num-gpus=16 --resources='{\"NPU\": 16}'"
        exit 1
    fi
fi

echo "=========================================="
echo "Ray pre-flight check complete"
echo "==========================================="


# ===== additonal HCCL set =====
# export HCCL_SOCKET_IFNAME=enx00e04c683d6a
# export GLOO_SOCKET_IFNAME=enx00e04c683d6a
# HCCL_SOCKET_IFNAME=ens1f3
# export HCCL_INTRA_ROCE_ENABLE=1
# export HCCL_INTRA_PCIE_ENABLE=0


# ===== user-adjustable: paths =====
# POLICY_PATH: HF dir of the policy (weights required).
# DRAFT_PATH : HF dir of the EAGLE3 draft (config.json required; *.safetensors
#              optional -- if absent the draft starts from random init, which is
#              fine for online training).
# Filled for the on-box Qwen3-32B (dense) + its converted EAGLE3 draft.
# Policy: 62G, 17 shards, Qwen3ForCausalLM (dense, hidden=5120, 40 layers).
# Draft : converted from the speculators-format ckpt via
#   scripts/converter_speculators_to_verl_eagle3.py  (full key match, no random init).
# MoE target: Qwen3-30B-A3B (128 experts, top-8, hidden 2048, 48 layers).
#   policy = the 57G MoE body (Qwen3MoeForCausalLM); draft = the 278M EAGLE3 draft
#   (already in verl format: LlamaForCausalLMEagle3, midlayer.* keys, t2d/d2t
#   buffers baked in -- NO converter needed).
# /mnt/weight is an NFS(RDMA) autofs mount; verified reachable at launch time.
POLICY_PATH=${POLICY_PATH:-/home/weight/Qwen3-8B}
DRAFT_PATH=${DRAFT_PATH:-/home/weight/qwen3_8b_eagle3}

# ===== user-adjustable: hardware (Qwen3-8B Dense: TP4 on 16 cards) =====
# Dense model (non-MoE): EP=1 (no expert parallelism needed)
# 16 cards: TP=4, DP=4 (16 / 4 = 4)
# Total GPUs = TP × PP × EP × DP = 4 × 1 × 1 × 4 = 16
NNODES=${NNODES:-1}
NPUS_PER_NODE=${NPUS_PER_NODE:-16}
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}

# ===== user-adjustable: EAGLE3 draft training =====
DRAFT_ENABLE_TRAIN=${DRAFT_ENABLE_TRAIN:-True}
EAGLE3_ENABLE_ROLLOUT=${EAGLE3_ENABLE_ROLLOUT:-True}    # rollout-side speculative decoding ON by default
TTT_LENGTH=${TTT_LENGTH:-1}                              # 1 = single-step (P2a); >1 = autoregressive TTT (P2b)
# Draft implementation backend:
#   False (default) = plain nn.Module (LlamaForCausalLMEagle3), REPLICATED, DDP over DP
#                     group. Stable path. No TP -> full 278M params + AdamW state per card.
#   True            = MegatronModule (MegatronEagle3DraftModel), TransformerBlock is
#                     TP-sharded like the policy -> ~2.2GB/card less at TP=4. Requires the
#                     draft build + HF->Megatron weight map to be complete (in progress).
USE_MEGATRON_DRAFT=${USE_MEGATRON_DRAFT:-True}
# Draft TP degree (only when USE_MEGATRON_DRAFT=True). Controls how many ways the draft
# backbone (qkv/proj/mlp/lm_head) is tensor-parallel sharded:
#   0 (default) = auto: follow policy TP (TRAIN_TP), auto-capped to a divisor of the
#                 draft's num_query_groups (=4 here) so the QKV fusion stays valid
#                 -> at TRAIN_TP=4 the draft shards 4 ways (full ~2.2GB/card saving).
#   <n>         = force draft TP = n (must divide BOTH policy TP and num_query_groups);
#                 n < policy_tp builds a dedicated draft TP sub-group (PP=1/CP=1 only).
DRAFT_TP=${DRAFT_TP:-4}  #2
DRAFT_LR=${DRAFT_LR:-1e-4}
LOSS_WEIGHT=${LOSS_WEIGHT:-1.0}
NUM_SPEC_TOKENS=${NUM_SPEC_TOKENS:-3}
# This draft IS vocab-compressed: draft_vocab_size=32000 vs policy vocab 151936.
# The loss side (loss_mcore.py:filter_teacher_to_draft_vocab) uses t2d to squeeze
# the 151936-d teacher logits down to the draft's 32000-d head, so compression
# MUST be enabled or the dims won't line up. The t2d/d2t mapping ships inside the
# converted ckpt (safetensors buffers), so VOCAB_MAPPING_PATH stays empty --
# build_draft_module loads the ckpt-embedded mapping automatically.
ENABLE_VOCAB_COMPRESSION=${ENABLE_VOCAB_COMPRESSION:-True}
DRAFT_VOCAB_SIZE=${DRAFT_VOCAB_SIZE:-32000}
VOCAB_MAPPING_PATH=${VOCAB_MAPPING_PATH:-""}

# ===== user-adjustable: data / batch / schedule =====
TRAIN_FILE=${TRAIN_FILE:-/home/t00972278/dataset/dapo-math-17k/dapo-math-17k.parquet}
VAL_FILE=${VAL_FILE:-/home/t00972278/dataset/dapo-math-17k/test.parquet}
max_prompt_length=${MAX_PROMPT_LENGTH:-512}
max_response_length=${MAX_RESPONSE_LENGTH:-8192}  # OOM: 长序列 → gather 峰值高
# max_response_length=${MAX_RESPONSE_LENGTH:-1024}    # 降到 1024 减少显存峰值


# ====batch-size====
train_prompt_bsz=${TRAIN_BATCH_SIZE:-4}            #代表的是
train_prompt_mini_bsz=${PPO_MINI_BATCH_SIZE:-1}     
ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU=${ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU:-1}


n_resp_per_prompt=${ROLLOUT_N:-8}
actor_lr=${ACTOR_LR:-1e-6}
total_epochs=${TOTAL_EPOCHS:-1}
total_training_steps=${TOTAL_TRAINING_STEPS:-100}
save_freq=${SAVE_FREQ:-10}
test_freq=${TEST_FREQ:--1}
ppo_max_token_len=$((max_prompt_length + max_response_length))

# ===== parallelism for Qwen3-8B Dense model =====
# Dense model (non-MoE): no EP/ETP needed. PP=1 still required by EAGLE3 gate.
# 16 cards: TP=4, DP=4 implied (16 / 4 = 4).
# SP (Sequence Parallel): For dense models with TP>1, SP is optional but recommended
# to reduce activation memory. SP size automatically equals TP size.
train_tp=${TRAIN_TP:-4}
train_pp=1
train_cp=1
train_sp=${TRAIN_SP:-True}   # Sequence Parallel: recommended for TP>1 to save memory
train_ep=1                     # Dense model: EP must be 1 (no expert parallelism)
train_etp=1                    # Dense model: ETP must be 1
gen_tp=${GEN_TP:-4}

project_name=${PROJECT_NAME:-verl_eagle3}
experiment_name=${EXPERIMENT_NAME:-qwen3_eagle3_megatron}
CKPTS_DIR=${CKPTS_DIR:-"/home/t00972278/verl/ckpts/${project_name}/${experiment_name}"}

########################### parameter arrays ###########################

DATA=(
    data.train_files="['$TRAIN_FILE']"
    data.val_files="['$VAL_FILE']"
    data.prompt_key=prompt
    data.train_batch_size=${train_prompt_bsz}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=False
    data.truncation='left'
    data.trust_remote_code=True
)

# POLICY path + EAGLE3 draft config (draft is nested UNDER model, path is separate)
MODEL=(
    actor_rollout_ref.model.path="${POLICY_PATH}"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.trust_remote_code=True
    # ---- EAGLE3 (all fields; see verl/workers/config/model.py Eagle3Config) ----
    actor_rollout_ref.model.eagle3.enable_train=${DRAFT_ENABLE_TRAIN}
    actor_rollout_ref.model.eagle3.enable_rollout=${EAGLE3_ENABLE_ROLLOUT}
    actor_rollout_ref.model.eagle3.draft_model_path="${DRAFT_PATH}"
    actor_rollout_ref.model.eagle3.loss_weight=${LOSS_WEIGHT}
    actor_rollout_ref.model.eagle3.capture_layer_ids=[]
    actor_rollout_ref.model.eagle3.method=eagle3
    actor_rollout_ref.model.eagle3.num_speculative_tokens=${NUM_SPEC_TOKENS}
    actor_rollout_ref.model.eagle3.draft_optim_lr=${DRAFT_LR}
    actor_rollout_ref.model.eagle3.draft_optim_weight_decay=0.01
    actor_rollout_ref.model.eagle3.draft_optim_clip_grad=1.0
    # actor_rollout_ref.model.eagle3.draft_optim_offload=${DRAFT_OPTIM_OFFLOAD:-True}  # 泄漏：每步累积 optimizer 状态
    actor_rollout_ref.model.eagle3.draft_optim_offload=${DRAFT_OPTIM_OFFLOAD:-False}  # 关闭 offload，常驻 NPU (~2.8GB)
    # actor_rollout_ref.model.eagle3.draft_forward_checkpoint=${DRAFT_FWD_CKPT:-False}  # OOM: 中间激活占显存
    actor_rollout_ref.model.eagle3.draft_forward_checkpoint=${DRAFT_FWD_CKPT:-True}    # 开启重计算省显存
    actor_rollout_ref.model.eagle3.enable_vocab_compression=${ENABLE_VOCAB_COMPRESSION}
    actor_rollout_ref.model.eagle3.draft_vocab_size=${DRAFT_VOCAB_SIZE}
    actor_rollout_ref.model.eagle3.vocab_mapping_path="${VOCAB_MAPPING_PATH}"
    actor_rollout_ref.model.eagle3.backend=megatron
    actor_rollout_ref.model.eagle3.ttt_length=${TTT_LENGTH}
    actor_rollout_ref.model.eagle3.draft_pipeline_stage=last
    actor_rollout_ref.model.eagle3.allow_pp_gt_1=False
    actor_rollout_ref.model.eagle3.use_megatron_draft=${USE_MEGATRON_DRAFT}
    actor_rollout_ref.model.eagle3.draft_tensor_parallel_size=${DRAFT_TP}
)

ALGORITHM=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    algorithm.kl_ctrl.kl_coef=0.0
)

# NOTE: actor.optim.lr is the POLICY lr; the draft has its OWN optimizer/lr
# (eagle3.draft_optim_lr above). They are deliberately independent (grad isolation).
ACTOR=(
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.use_dynamic_bsz=False
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.ppo_epochs=1
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len}
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz}

    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU}

    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${train_tp}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${train_pp}
    actor_rollout_ref.actor.megatron.context_parallel_size=${train_cp}
    actor_rollout_ref.actor.megatron.sequence_parallel=${train_sp}
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${train_ep}
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${train_etp}
    actor_rollout_ref.actor.megatron.param_offload=True
    actor_rollout_ref.actor.megatron.grad_offload=True
    actor_rollout_ref.actor.megatron.optimizer_offload=True
    actor_rollout_ref.actor.megatron.use_mbridge=True
    # vanilla_mbridge=True selects the installed third-party `mbridge` package.
    # Default (False) would import NVIDIA `megatron.bridge`, which is NOT installed
    # on this Ascend box -> ModuleNotFoundError at engine.initialize(). All Ascend
    # examples use vanilla_mbridge=True (deprecation warning is cosmetic).
    actor_rollout_ref.actor.megatron.vanilla_mbridge=True
    # recompute set
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1


    # profile
    actor_rollout_ref.actor.profiler.enable=True
    actor_rollout_ref.actor.profiler.all_ranks=False
    actor_rollout_ref.actor.profiler.ranks="[0]" # 只采集rank0
    actor_rollout_ref.actor.profiler.tool_config.npu.discrete=True # 推荐使用离散模式，各阶段数据分开存储
    actor_rollout_ref.actor.profiler.tool_config.npu.contents="['npu','cpu']" # 控制采集列表，默认cpu、npu，可配置memory、shapes、module等
    actor_rollout_ref.actor.profiler.tool_config.npu.level=level1
    actor_rollout_ref.actor.profiler.tool_config.npu.analysis=False # 禁用自动数据解析
    )

REF=(
    actor_rollout_ref.ref.use_torch_compile=False
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${ppo_max_token_len}
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${train_tp}
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${train_pp}
    actor_rollout_ref.ref.megatron.context_parallel_size=${train_cp}
    actor_rollout_ref.ref.megatron.sequence_parallel=${train_sp}
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${train_ep}
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${train_etp}
    actor_rollout_ref.ref.megatron.param_offload=True
    actor_rollout_ref.ref.megatron.use_mbridge=True
    actor_rollout_ref.ref.megatron.vanilla_mbridge=True
)

ROLLOUT=(
    # vLLM inference backend (Ascend uses vllm_ascend; sglang is not installed here).
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.n=${n_resp_per_prompt}
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.top_k=-1
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7    # 0.5
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp}
    actor_rollout_ref.rollout.enforce_eager=${ENFORCE_EAGER:-False}
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len}
    actor_rollout_ref.rollout.val_kwargs.n=1
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0
    # ===== rollout-phase NPU profiling (trace policy verify forward for cross-check) =====
    # Agent Loop mode REQUIRES discrete=True; rollout ranks = Replica Rank (推理实例索引).
    # vLLM engine auto-captures AsyncLLM scheduling + inference process; analysis 不支持在线，需离线用 MindStudio Insight 解析。
    actor_rollout_ref.rollout.profiler.enable=${PROFILE_ROLLOUT:-True}
    actor_rollout_ref.rollout.profiler.all_ranks=False
    actor_rollout_ref.rollout.profiler.ranks="[0]"
    actor_rollout_ref.rollout.profiler.tool_config.npu.discrete=True
    actor_rollout_ref.rollout.profiler.tool_config.npu.contents="['npu','cpu']"
    actor_rollout_ref.rollout.profiler.tool_config.npu.profile_token_start=30
    actor_rollout_ref.rollout.profiler.tool_config.npu.profile_token_end=60
)

# Metrics persistence: console (stdout) + file (JSONL, one line/step) + tensorboard.
#   file    -> ${VERL_FILE_LOGGER_ROOT}/${project_name}/${experiment_name}.jsonl
#   tensorboard -> ${TENSORBOARD_DIR:-tensorboard_log/${project_name}/${experiment_name}}
# The 5 EAGLE3 metrics land under keys: timing_s/gen, perf/throughput,
# critic/rewards/mean, rollout/spec_accept_length, actor/draft_loss.
export VERL_FILE_LOGGER_ROOT="/home/t00972278/desk/eagle3_train/eagle3_result/logs/metrics"
export TENSORBOARD_DIR="/home/t00972278/desk/eagle3_train/eagle3_result/logs/tb/${experiment_name}"
# Force all three backends; do NOT allow env override (avoids accidentally logging only to console).
TRAINER_LOGGER='["console","file","tensorboard"]'

TRAINER=(
    trainer.logger="${TRAINER_LOGGER}"
    trainer.project_name="${project_name}"
    trainer.experiment_name="${experiment_name}"
    trainer.nnodes="${NNODES}"
    trainer.n_gpus_per_node="${NPUS_PER_NODE}"
    trainer.device='npu'
    trainer.val_before_train=False
    trainer.total_epochs=${total_epochs}
    trainer.total_training_steps=${total_training_steps}
    trainer.save_freq=${save_freq}
    trainer.test_freq=${test_freq}
    trainer.default_local_dir="${CKPTS_DIR}"
)

# ===== NPU profiler (global control) — trace rollout phase for 1-2 steps =====
# Traces saved under save_path; open with MindStudio Insight (offline analysis).
# Disable by setting PROFILE_STEPS=null.
PROFILE_STEPS=${PROFILE_STEPS:-"[2,3]"}
PROFILE_SAVE_PATH=${PROFILE_SAVE_PATH:-/home/t00972278/desk/eagle3_train/eagle3_result/profile}
PROFILER=(
    global_profiler.tool=npu
    global_profiler.steps="${PROFILE_STEPS}"
    global_profiler.save_path="${PROFILE_SAVE_PATH}"
)

########################### launch ###########################
# EAGLE3 draft training metrics to watch (see eagle3开发设计/verl_eagle3.md §5.6):
#   train : draft_loss (produced), per-step loss (TODO), draft top-1 acc (TODO)
#   rollout (only when enable_rollout=True + refit): rollout/spec_accept_rate,
#           rollout/spec_accept_length; end-to-end speedup derived from those.
python3 -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name='ppo_megatron_trainer.yaml' \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ALGORITHM[@]}" \
    "${ACTOR[@]}" \
    "${REF[@]}" \
    "${ROLLOUT[@]}" \
    "${TRAINER[@]}" \
    "${PROFILER[@]}" \
    "$@"\




