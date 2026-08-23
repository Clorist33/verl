unset http_proxy
unset https_proxy

export VLLM_USE_V1=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=1024
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_ENABLE_NZ=2
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

python -m vllm.entrypoints.openai.api_server \
    --model /home/weight/Qwen3-30B-A3B \
    --served-model-name qwen3 \
    --trust-remote-code \
    --max-num-seqs 16 \
    --max-model-len 135000 \
    --max-num-batched-tokens 32000 \
    --tensor-parallel-size 4 \
    --data-parallel-size 1 \
    --enable-expert-parallel \
    --port 1999 \
    --distributed_executor_backend "mp" \
    --no-enable-prefix-caching \
    --async-scheduling True \
    --quantization ascend \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --gpu-memory-utilization 0.85  \
    --speculative-config '{"method": "eagle3","model": "/home/weight/Qwen3-a3B_eagle3", "num_speculative_tokens": 3}' \

