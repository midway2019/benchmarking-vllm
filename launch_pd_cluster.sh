#!/bin/bash
# launch_pd_cluster.sh
# 启动 SGLang 1P3D PD分离集群 + sglang_router
# 1 Prefill (GPU0) + 3 Decode (GPU1/2/3)，使用 NIXL 传输后端

set -e

MODEL_NAME="${HF_MODEL_NAME:-/artesia-workspace/models/Qwen3-8B}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
MEM_FRACTION="${MEM_FRACTION:-0.85}"

PREFILL_PORT=29100
DECODE_PORTS=(29201 29202 29203)
ROUTER_PORT=29001

LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo " SGLang PD Disaggregated Cluster Launcher"
echo "=========================================="
echo "Model:          $MODEL_NAME"
echo "Prefill Port:   $PREFILL_PORT (GPU 0)"
echo "Decode Ports:   ${DECODE_PORTS[*]} (GPU 1/2/3)"
echo "Router Port:    $ROUTER_PORT"
echo "Transfer:       NIXL (NVLink)"
echo "Radix Cache:    DISABLED"
echo "=========================================="

trap 'cleanup' INT TERM EXIT

cleanup() {
    echo ""
    echo "Cleaning up all SGLang processes..."
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
        if curl -s "http://localhost:${port}/v1/models" > /dev/null 2>&1 || \
           curl -s "http://localhost:${port}/health" > /dev/null 2>&1; then
            echo "$name (port $port) is ready! (${elapsed}s)"
            return 0
        fi
        sleep 3
        elapsed=$((elapsed + 3))
    done
    echo "ERROR: $name (port $port) failed to start within ${max_wait}s"
    return 1
}

# --- Launch Prefill Instance (GPU 0) ---
echo ""
echo "[1/5] Launching Prefill instance on GPU 0 (port $PREFILL_PORT)..."
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
    --model-path "$MODEL_NAME" \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend nixl \
    --host 0.0.0.0 \
    --port $PREFILL_PORT \
    --mem-fraction-static $MEM_FRACTION \
    --disable-radix-cache \
    --trust-remote-code \
    > "$LOG_DIR/prefill.log" 2>&1 &
PREFILL_PID=$!
echo "  Prefill PID: $PREFILL_PID"

# --- Launch Decode Instance 1 (GPU 1) ---
echo "[2/5] Launching Decode instance 1 on GPU 1 (port ${DECODE_PORTS[0]})..."
CUDA_VISIBLE_DEVICES=1 python -m sglang.launch_server \
    --model-path "$MODEL_NAME" \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend nixl \
    --host 0.0.0.0 \
    --port ${DECODE_PORTS[0]} \
    --mem-fraction-static $MEM_FRACTION \
    --disable-radix-cache \
    --trust-remote-code \
    > "$LOG_DIR/decode_1.log" 2>&1 &
DECODE1_PID=$!
echo "  Decode 1 PID: $DECODE1_PID"

# --- Launch Decode Instance 2 (GPU 2) ---
echo "[3/5] Launching Decode instance 2 on GPU 2 (port ${DECODE_PORTS[1]})..."
CUDA_VISIBLE_DEVICES=2 python -m sglang.launch_server \
    --model-path "$MODEL_NAME" \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend nixl \
    --host 0.0.0.0 \
    --port ${DECODE_PORTS[1]} \
    --mem-fraction-static $MEM_FRACTION \
    --disable-radix-cache \
    --trust-remote-code \
    > "$LOG_DIR/decode_2.log" 2>&1 &
DECODE2_PID=$!
echo "  Decode 2 PID: $DECODE2_PID"

# --- Launch Decode Instance 3 (GPU 3) ---
echo "[4/5] Launching Decode instance 3 on GPU 3 (port ${DECODE_PORTS[2]})..."
CUDA_VISIBLE_DEVICES=3 python -m sglang.launch_server \
    --model-path "$MODEL_NAME" \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend nixl \
    --host 0.0.0.0 \
    --port ${DECODE_PORTS[2]} \
    --mem-fraction-static $MEM_FRACTION \
    --disable-radix-cache \
    --trust-remote-code \
    > "$LOG_DIR/decode_3.log" 2>&1 &
DECODE3_PID=$!
echo "  Decode 3 PID: $DECODE3_PID"

# --- Wait for all instances ---
echo ""
echo "Waiting for all instances to be ready..."
wait_for_server $PREFILL_PORT "Prefill"
wait_for_server ${DECODE_PORTS[0]} "Decode-1"
wait_for_server ${DECODE_PORTS[1]} "Decode-2"
wait_for_server ${DECODE_PORTS[2]} "Decode-3"

# --- Launch Simple Router ---
echo ""
echo "[5/5] Launching simple_router (port $ROUTER_PORT)..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "$SCRIPT_DIR/simple_router.py" \
    --prefill http://127.0.0.1:$PREFILL_PORT \
    --decode http://127.0.0.1:${DECODE_PORTS[0]} http://127.0.0.1:${DECODE_PORTS[1]} http://127.0.0.1:${DECODE_PORTS[2]} \
    --host 0.0.0.0 \
    --port $ROUTER_PORT \
    > "$LOG_DIR/router.log" 2>&1 &
ROUTER_PID=$!
echo "  Router PID: $ROUTER_PID"

# Wait for router
sleep 5
echo ""
echo "=========================================="
echo " All instances and router are ready!"
echo "=========================================="
echo "  Prefill:  http://localhost:$PREFILL_PORT"
echo "  Decode-1: http://localhost:${DECODE_PORTS[0]}"
echo "  Decode-2: http://localhost:${DECODE_PORTS[1]}"
echo "  Decode-3: http://localhost:${DECODE_PORTS[2]}"
echo "  Router:   http://localhost:$ROUTER_PORT"
echo "=========================================="

# Keep running until interrupted
wait
