#!/bin/bash
# launch_pd_cluster.sh
# 启动 1P3D vLLM PD分离集群：1 Prefill (GPU0) + 3 Decode (GPU1/2/3)
# 使用 P2pNcclConnector，关闭 prefix caching

set -e

MODEL_NAME="${HF_MODEL_NAME:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
VLLM_HOST_IP="${VLLM_HOST_IP:-127.0.0.1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"

PREFILL_PORT=8100
DECODE_PORTS=(8201 8202 8203)
PROXY_PORT=30001

LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo " vLLM PD Disaggregated Cluster Launcher"
echo "=========================================="
echo "Model:          $MODEL_NAME"
echo "Host IP:        $VLLM_HOST_IP"
echo "Max Model Len:  $MAX_MODEL_LEN"
echo "GPU Mem Util:   $GPU_MEM_UTIL"
echo "Prefill Port:   $PREFILL_PORT (GPU 0)"
echo "Decode Ports:   ${DECODE_PORTS[*]} (GPU 1/2/3)"
echo "=========================================="

trap 'cleanup' INT TERM EXIT

cleanup() {
    echo ""
    echo "Cleaning up all vLLM processes..."
    kill $(jobs -p) 2>/dev/null || true
    wait 2>/dev/null || true
    echo "Cleanup complete."
}

wait_for_server() {
    local port=$1
    local name=$2
    local max_wait=600
    local elapsed=0
    echo "Waiting for $name (port $port) to be ready..."
    while [ $elapsed -lt $max_wait ]; do
        if curl -s "http://localhost:${port}/v1/models" > /dev/null 2>&1; then
            echo "$name (port $port) is ready! (${elapsed}s)"
            return 0
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done
    echo "ERROR: $name (port $port) failed to start within ${max_wait}s"
    return 1
}

# --- Launch Prefill Instance (GPU 0, kv_producer) ---
echo ""
echo "[1/4] Launching Prefill instance on GPU 0 (port $PREFILL_PORT)..."
CUDA_VISIBLE_DEVICES=0 vllm serve "$MODEL_NAME" \
    --host 0.0.0.0 \
    --port $PREFILL_PORT \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEM_UTIL \
    --no-enable-prefix-caching \
    --trust-remote-code \
    --kv-transfer-config \
    '{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":4,"kv_buffer_size":"1e9","kv_port":"14579","kv_connector_extra_config":{"proxy_ip":"'"$VLLM_HOST_IP"'","proxy_port":"'"$PROXY_PORT"'","http_ip":"'"$VLLM_HOST_IP"'","http_port":"'"$PREFILL_PORT"'","send_type":"PUT_ASYNC"}}' \
    > "$LOG_DIR/prefill.log" 2>&1 &
PREFILL_PID=$!
echo "  Prefill PID: $PREFILL_PID"

# --- Launch Decode Instance 1 (GPU 1, kv_consumer) ---
echo "[2/4] Launching Decode instance 1 on GPU 1 (port ${DECODE_PORTS[0]})..."
CUDA_VISIBLE_DEVICES=1 vllm serve "$MODEL_NAME" \
    --host 0.0.0.0 \
    --port ${DECODE_PORTS[0]} \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEM_UTIL \
    --no-enable-prefix-caching \
    --trust-remote-code \
    --kv-transfer-config \
    '{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":4,"kv_buffer_size":"1e10","kv_port":"14580","kv_connector_extra_config":{"proxy_ip":"'"$VLLM_HOST_IP"'","proxy_port":"'"$PROXY_PORT"'","http_ip":"'"$VLLM_HOST_IP"'","http_port":"'"${DECODE_PORTS[0]}"'","send_type":"PUT_ASYNC"}}' \
    > "$LOG_DIR/decode_1.log" 2>&1 &
DECODE1_PID=$!
echo "  Decode 1 PID: $DECODE1_PID"

# --- Launch Decode Instance 2 (GPU 2, kv_consumer) ---
echo "[3/4] Launching Decode instance 2 on GPU 2 (port ${DECODE_PORTS[1]})..."
CUDA_VISIBLE_DEVICES=2 vllm serve "$MODEL_NAME" \
    --host 0.0.0.0 \
    --port ${DECODE_PORTS[1]} \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEM_UTIL \
    --no-enable-prefix-caching \
    --trust-remote-code \
    --kv-transfer-config \
    '{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_rank":2,"kv_parallel_size":4,"kv_buffer_size":"1e10","kv_port":"14581","kv_connector_extra_config":{"proxy_ip":"'"$VLLM_HOST_IP"'","proxy_port":"'"$PROXY_PORT"'","http_ip":"'"$VLLM_HOST_IP"'","http_port":"'"${DECODE_PORTS[1]}"'","send_type":"PUT_ASYNC"}}' \
    > "$LOG_DIR/decode_2.log" 2>&1 &
DECODE2_PID=$!
echo "  Decode 2 PID: $DECODE2_PID"

# --- Launch Decode Instance 3 (GPU 3, kv_consumer) ---
echo "[4/4] Launching Decode instance 3 on GPU 3 (port ${DECODE_PORTS[2]})..."
CUDA_VISIBLE_DEVICES=3 vllm serve "$MODEL_NAME" \
    --host 0.0.0.0 \
    --port ${DECODE_PORTS[2]} \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEM_UTIL \
    --no-enable-prefix-caching \
    --trust-remote-code \
    --kv-transfer-config \
    '{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_rank":3,"kv_parallel_size":4,"kv_buffer_size":"1e10","kv_port":"14582","kv_connector_extra_config":{"proxy_ip":"'"$VLLM_HOST_IP"'","proxy_port":"'"$PROXY_PORT"'","http_ip":"'"$VLLM_HOST_IP"'","http_port":"'"${DECODE_PORTS[2]}"'","send_type":"PUT_ASYNC"}}' \
    > "$LOG_DIR/decode_3.log" 2>&1 &
DECODE3_PID=$!
echo "  Decode 3 PID: $DECODE3_PID"

# --- Wait for all instances to be ready ---
echo ""
echo "Waiting for all instances to be ready..."
wait_for_server $PREFILL_PORT "Prefill"
wait_for_server ${DECODE_PORTS[0]} "Decode-1"
wait_for_server ${DECODE_PORTS[1]} "Decode-2"
wait_for_server ${DECODE_PORTS[2]} "Decode-3"

echo ""
echo "=========================================="
echo " All instances are ready!"
echo "=========================================="
echo "  Prefill:  http://localhost:$PREFILL_PORT"
echo "  Decode-1: http://localhost:${DECODE_PORTS[0]}"
echo "  Decode-2: http://localhost:${DECODE_PORTS[1]}"
echo "  Decode-3: http://localhost:${DECODE_PORTS[2]}"
echo "=========================================="

# Keep running until interrupted
wait
