#!/usr/bin/env bash
# EAGLE3 纯推理 server —— 参照 eagle3_rollout.sh，配置对齐训练脚本 run_qwen3_eagle3_megatron.sh 的 rollout 配置
# 差异点（相对 eagle3_rollout.sh）：
#   - model 换成训练的 policy（bf16 非量化版），去掉 --quantization
#   - draft 换成训练用的 /home/weight/Qwen3-a3B_eagle3
#   - gpu-memory-utilization 0.7（训练 rollout 配置）
#   - 保留 prefix caching（训练 rollout 默认开启）
#   - max-num-batched-tokens 8192（训练 rollout 的 chunked prefill 配置）
#
# 用法：
#   bash eagle3_inference_server.sh          # 起 server（端口 1999）
#
# 测试（另开终端发请求）：
#   curl http://localhost:1999/v1/completions \
#     -H "Content-Type: application/json" \
#     -d '{"model": "qwen3", "prompt": "Solve: 1+1=", "max_tokens": 2048, "temperature": 1.0}'
#
# 看接受率：vLLM 默认每 10s 打印一次统计日志，发请求后在 server 日志中看
#   "SpecDecoding metrics" 行，包含 Draft acceptance rate / Mean acceptance length

unset http_proxy
unset https_proxy

# ===== 环境变量（对齐训练脚本）=====
export VLLM_USE_V1=1
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
export HCCL_CONNECT_TIMEOUT=1500
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=1024
export TASK_QUEUE_ENABLE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export DISABLE_L2_CACHE=1
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_ENABLE_NZ=0
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# ===== 模型路径（训练配置）=====
POLICY_PATH=${POLICY_PATH:-/home/weight/Qwen3-8B}   # /home/weight/Qwen3-30B-A3B
DRAFT_PATH=${DRAFT_PATH:-/home/weight/qwen3_8b_eagle3}  # /home/weight/Qwen3-30B-A3B-eagle-model
NUM_SPEC_TOKENS=${NUM_SPEC_TOKENS:-3}

python -m vllm.entrypoints.openai.api_server \
    --model ${POLICY_PATH} \
    --served-model-name qwen3-8B \
    --trust-remote-code \
    --max-num-seqs 256 \
    --max-model-len 9216 \
    --max-num-batched-tokens 8192 \
    --tensor-parallel-size 4 \
    --data-parallel-size 4 \
    --port 9090 \
    --enable-chunked-prefill \
    --async-scheduling True \
    --profiler-config '{"profiler": "torch", "torch_profiler_dir": "/home/t00972278/profiling", "torch_profiler_with_stack": false}' \
    --gpu-memory-utilization 0.7 \
    --speculative-config "{\"method\": \"eagle3\",\"model\": \"${DRAFT_PATH}\", \"num_speculative_tokens\": ${NUM_SPEC_TOKENS}}"



# --enable-expert-parallel \
# --distributed_executor_backend "mp" \


