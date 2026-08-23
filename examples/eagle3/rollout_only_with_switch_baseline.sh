#!/usr/bin/env bash
# EAGLE3 Rollout-Only 测试脚本（使用真正的 rollout_only 开关）
# 用于测试 EAGLE3 投机推理性能和接受率

set -xeuo pipefail

# ===== 环境变量 =====
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
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
export VLLM_ASCEND_ENABLE_NZ=0
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# ===== Ray 集群启动 =====
echo "=========================================="
echo "Starting Ray cluster with 16 GPUs"
echo "=========================================="

RAY_RUNNING=false
if ray status >/dev/null 2>&1; then
    CURRENT_GPUS=$(python3 -c "import ray; ray.init(address='auto'); g=ray.cluster_resources().get('GPU',0); ray.shutdown(); print(g)" 2>/dev/null || echo "0")
    if [ "$CURRENT_GPUS" = "16.0" ]; then
        echo "✓ Ray cluster already running with 16 GPUs"
        RAY_RUNNING=true
    else
        echo "✗ Restarting Ray with correct configuration..."
        ray stop --force 2>/dev/null || true
        sleep 3
    fi
fi

if [ "$RAY_RUNNING" = false ]; then
    ray start --head --num-gpus=16 --resources='{"NPU": 16}'
    sleep 2
fi

# ===== 模型路径 =====
POLICY_PATH=${POLICY_PATH:-/home/weight/Qwen3-8B}
DRAFT_PATH=${DRAFT_PATH:-/home/weight/qwen3_8b_eagle3}

# ===== EAGLE3 配置 =====
DRAFT_ENABLE_TRAIN=${DRAFT_ENABLE_TRAIN:-False}  # 不训练 draft
EAGLE3_ENABLE_ROLLOUT=${EAGLE3_ENABLE_ROLLOUT:-False}  # 启用 rollout 投机推理
NUM_SPEC_TOKENS=${NUM_SPEC_TOKENS:-3}

# ===== 关键修复：Draft TP 对齐 =====
DRAFT_TP=${DRAFT_TP:-4}  # 修复：设为 4，与 policy TP 一致
USE_MEGATRON_DRAFT=${USE_MEGATRON_DRAFT:-True}

# ===== 硬件配置 =====
train_tp=${TRAIN_TP:-4}
train_pp=1
train_ep=1
train_etp=1
gen_tp=${GEN_TP:-4}

# ===== 数据配置（rollout-only 模式只需要少量数据） =====
TRAIN_FILE=${TRAIN_FILE:-/home/t00972278/dataset/dapo-math-17k/dapo-math-17k.parquet}
max_prompt_length=${MAX_PROMPT_LENGTH:-512}
max_response_length=${MAX_RESPONSE_LENGTH:-8192}
train_prompt_bsz=${TRAIN_BATCH_SIZE:-4}
n_resp_per_prompt=${ROLLOUT_N:-8}

# ===== Rollout-Only 模式配置 =====
total_training_steps=${TOTAL_TRAINING_STEPS:-3}  # 跑 3 个 steps 来观察稳定性
total_epochs=${TOTAL_EPOCHS:-1}

# ===== 输出目录 =====
project_name=${PROJECT_NAME:-verl_eagle3_rollout_only}
experiment_name=${EXPERIMENT_NAME:-qwen3_eagle3_rollout_test_$(date +%Y%m%d_%H%M%S)}
CKPTS_DIR="/home/t00972278/verl/ckpts/${project_name}/${experiment_name}"

# ===== 日志目录 =====
export VERL_FILE_LOGGER_ROOT="/home/t00972278/desk/eagle3_rollout/rollout_only_logs/metrics"
export TENSORBOARD_DIR="/home/t00972278/desk/eagle3_rollout/rollout_only_logs/tb/${experiment_name}"

echo "=========================================="
echo "EAGLE3 Rollout-Only Test Configuration"
echo "=========================================="
echo "✅ rollout_only: TRUE (真正的开关)"
echo "Policy Model: ${POLICY_PATH}"
echo "Draft Model: ${DRAFT_PATH}"
echo "EAGLE3 Rollout: ${EAGLE3_ENABLE_ROLLOUT}"
echo "Draft TP: ${DRAFT_TP} (修复：对齐 policy TP)"
echo "Policy TP: ${train_tp}"
echo "Num Spec Tokens: ${NUM_SPEC_TOKENS}"
echo "Total Steps: ${total_training_steps}"
echo "Batch Size: ${train_prompt_bsz}"
echo "Responses per Prompt: ${n_resp_per_prompt}"
echo "Logs: ${VERL_FILE_LOGGER_ROOT}"
echo "=========================================="

# ===== 参数数组 =====
DATA=(
    data.train_files="['$TRAIN_FILE']"
    data.val_files="['$TRAIN_FILE']"
    data.prompt_key=prompt
    data.train_batch_size=${train_prompt_bsz}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=False
    data.truncation='left'
    data.trust_remote_code=True
)

MODEL=(
    actor_rollout_ref.model.path="${POLICY_PATH}"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.trust_remote_code=True
    # EAGLE3 配置
    actor_rollout_ref.model.eagle3.enable_train=${DRAFT_ENABLE_TRAIN}
    actor_rollout_ref.model.eagle3.enable_rollout=${EAGLE3_ENABLE_ROLLOUT}
    actor_rollout_ref.model.eagle3.draft_model_path="${DRAFT_PATH}"
    actor_rollout_ref.model.eagle3.loss_weight=1.0
    actor_rollout_ref.model.eagle3.capture_layer_ids=[]
    actor_rollout_ref.model.eagle3.method=eagle3
    actor_rollout_ref.model.eagle3.num_speculative_tokens=${NUM_SPEC_TOKENS}
    actor_rollout_ref.model.eagle3.draft_optim_lr=1e-4
    actor_rollout_ref.model.eagle3.draft_optim_weight_decay=0.01
    actor_rollout_ref.model.eagle3.draft_optim_clip_grad=1.0
    actor_rollout_ref.model.eagle3.draft_optim_offload=False
    actor_rollout_ref.model.eagle3.draft_forward_checkpoint=False
    actor_rollout_ref.model.eagle3.enable_vocab_compression=True
    actor_rollout_ref.model.eagle3.draft_vocab_size=32000
    actor_rollout_ref.model.eagle3.vocab_mapping_path=""
    actor_rollout_ref.model.eagle3.backend=megatron
    actor_rollout_ref.model.eagle3.ttt_length=1
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

ACTOR=(
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.ppo_epochs=1
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((max_prompt_length + max_response_length))
    actor_rollout_ref.actor.ppo_mini_batch_size=1
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${train_tp}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${train_pp}
    actor_rollout_ref.actor.megatron.context_parallel_size=1
    actor_rollout_ref.actor.megatron.sequence_parallel=True
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${train_ep}
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${train_etp}
    actor_rollout_ref.actor.megatron.use_mbridge=True
    actor_rollout_ref.actor.megatron.vanilla_mbridge=True
)

REF=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((max_prompt_length + max_response_length))
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${train_tp}
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${train_pp}
    actor_rollout_ref.ref.megatron.context_parallel_size=1
    actor_rollout_ref.ref.megatron.sequence_parallel=True
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${train_ep}
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${train_etp}
    actor_rollout_ref.ref.megatron.use_mbridge=True
    actor_rollout_ref.ref.megatron.vanilla_mbridge=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.n=${n_resp_per_prompt}
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.top_k=-1
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp}
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.max_num_batched_tokens=8192
    actor_rollout_ref.rollout.max_model_len=32768
    actor_rollout_ref.rollout.max_num_seqs=16
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((max_prompt_length + max_response_length))
)

TRAINER=(
    trainer.logger='["console","file"]'
    trainer.project_name="${project_name}"
    trainer.experiment_name="${experiment_name}"
    trainer.nnodes=1
    trainer.n_gpus_per_node=16
    trainer.device='npu'
    trainer.val_before_train=False
    trainer.rollout_only=True  # 🔥 真正的 rollout_only 开关
    trainer.total_epochs=${total_epochs}
    trainer.total_training_steps=${total_training_steps}
    trainer.save_freq=-1
    trainer.test_freq=-1
)

# ===== 启动训练（rollout_only=True） =====
echo "=========================================="
echo "Starting EAGLE3 Rollout-Only Test..."
echo "=========================================="

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
    "$@"

echo "=========================================="
echo "EAGLE3 Rollout-Only Test Completed!"
echo "=========================================="
echo "Check metrics at: ${VERL_FILE_LOGGER_ROOT}"
echo ""
echo "Key metrics to look for:"
echo "  - rollout/acceptance_rate (应该在 40-70%)"
echo "  - rollout/mean_acceptance_length (应该在 2.5-3.5)"
echo "  - timing_s/gen (生成时间，应该比修复前快 50%)"
echo "  - perf/throughput (吞吐量 tokens/s)"
echo ""
echo "应该看到日志："
echo "  [Rollout-Only Mode] Step X: Skipping training updates, only performing rollout"
echo "=========================================="
