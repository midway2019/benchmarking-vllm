#!/bin/bash
# run_benchmark.sh
# 一键运行 vLLM PD分离 Benchmark：启动集群 → 启动Proxy → 运行测试 → 生成报告 → 清理

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_NAME="${HF_MODEL_NAME:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
INPUT_LEN="${INPUT_LEN:-453}"
OUTPUT_LEN="${OUTPUT_LEN:-453}"
PROXY_PORT="${PROXY_PORT:-8000}"
RESULTS_DIR="${RESULTS_DIR:-results}"
NUM_WARMUP="${NUM_WARMUP:-3}"
INTER_BATCH_DELAY="${INTER_BATCH_DELAY:-5}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-600}"

LOG_DIR="./logs"
mkdir -p "$LOG_DIR" "$RESULTS_DIR"

CLUSTER_PID=""
PROXY_PID=""

echo "=========================================="
echo " vLLM PD Benchmark - Full Pipeline"
echo "=========================================="
echo "Model:        $MODEL_NAME"
echo "Input len:    $INPUT_LEN tokens"
echo "Output len:   $OUTPUT_LEN tokens"
echo "Proxy port:   $PROXY_PORT"
echo "Results dir:  $RESULTS_DIR"
echo "=========================================="

cleanup() {
    echo ""
    echo "Cleaning up..."
    if [ -n "$PROXY_PID" ] && kill -0 "$PROXY_PID" 2>/dev/null; then
        echo "  Stopping proxy server (PID $PROXY_PID)..."
        kill "$PROXY_PID" 2>/dev/null || true
    fi
    if [ -n "$CLUSTER_PID" ] && kill -0 "$CLUSTER_PID" 2>/dev/null; then
        echo "  Stopping PD cluster (PID $CLUSTER_PID)..."
        kill "$CLUSTER_PID" 2>/dev/null || true
    fi
    # Kill any remaining vllm processes started by this script
    pkill -f "launch_pd_cluster.sh" 2>/dev/null || true
    pkill -f "proxy_server.py" 2>/dev/null || true
    # Kill vllm serve processes on our ports
    for port in 8100 8201 8202 8203; do
        pid=$(lsof -ti:$port 2>/dev/null || true)
        if [ -n "$pid" ]; then
            echo "  Killing process on port $port (PID $pid)..."
            kill $pid 2>/dev/null || true
        fi
    done
    wait 2>/dev/null || true
    echo "Cleanup complete."
}

trap cleanup INT TERM EXIT

wait_for_server() {
    local port=$1
    local name=$2
    local max_wait=${3:-600}
    local elapsed=0
    echo "  Waiting for $name (port $port)..."
    while [ $elapsed -lt $max_wait ]; do
        if curl -s "http://localhost:${port}/health" > /dev/null 2>&1 || \
           curl -s "http://localhost:${port}/v1/models" > /dev/null 2>&1; then
            echo "  $name is ready! (${elapsed}s)"
            return 0
        fi
        sleep 3
        elapsed=$((elapsed + 3))
    done
    echo "  ERROR: $name failed to start within ${max_wait}s"
    return 1
}

# ========================================
# Step 1: Launch PD Cluster
# ========================================
echo ""
echo "[Step 1/5] Launching PD cluster (1P3D)..."
bash "$SCRIPT_DIR/launch_pd_cluster.sh" > "$LOG_DIR/cluster_launch.log" 2>&1 &
CLUSTER_PID=$!
echo "  Cluster launcher PID: $CLUSTER_PID"

# Wait for all vLLM instances
echo "  Waiting for vLLM instances to be ready (this may take a few minutes)..."
wait_for_server 8100 "Prefill" 600
wait_for_server 8201 "Decode-1" 600
wait_for_server 8202 "Decode-2" 600
wait_for_server 8203 "Decode-3" 600

echo "  All vLLM instances are ready!"

# ========================================
# Step 2: Launch Proxy Server
# ========================================
echo ""
echo "[Step 2/5] Launching proxy server on port $PROXY_PORT..."
python3 "$SCRIPT_DIR/proxy_server.py" --port "$PROXY_PORT" \
    > "$LOG_DIR/proxy.log" 2>&1 &
PROXY_PID=$!
echo "  Proxy PID: $PROXY_PID"

wait_for_server "$PROXY_PORT" "Proxy" 30

# ========================================
# Step 3: Run Benchmark
# ========================================
echo ""
echo "[Step 3/5] Running benchmark..."
echo "  This will test batch sizes: 16, 32, 48, ..., 384, 392"
echo "  Estimated time: depends on model and GPU performance"
echo ""

python3 "$SCRIPT_DIR/benchmark_pd.py" \
    --model "$MODEL_NAME" \
    --input-len "$INPUT_LEN" \
    --output-len "$OUTPUT_LEN" \
    --proxy-url "http://localhost:$PROXY_PORT" \
    --output "$RESULTS_DIR/benchmark_results.json" \
    --num-warmup "$NUM_WARMUP" \
    --timeout "$REQUEST_TIMEOUT" \
    --inter-batch-delay "$INTER_BATCH_DELAY" \
    2>&1 | tee "$LOG_DIR/benchmark.log"

BENCH_EXIT=$?
if [ $BENCH_EXIT -ne 0 ]; then
    echo "ERROR: Benchmark failed with exit code $BENCH_EXIT"
    echo "Check logs at: $LOG_DIR/benchmark.log"
    exit 1
fi

# ========================================
# Step 4: Generate Visualizations
# ========================================
echo ""
echo "[Step 4/5] Generating visualizations..."
python3 "$SCRIPT_DIR/visualize_results.py" \
    --input "$RESULTS_DIR/benchmark_results.json" \
    --output-dir "$RESULTS_DIR/plots" \
    2>&1 | tee "$LOG_DIR/visualize.log"

# ========================================
# Step 5: Summary
# ========================================
echo ""
echo "=========================================="
echo " Benchmark Complete!"
echo "=========================================="
echo "  Results JSON:  $RESULTS_DIR/benchmark_results.json"
echo "  Summary CSV:   $RESULTS_DIR/plots/benchmark_summary.csv"
echo "  Plots:         $RESULTS_DIR/plots/"
echo "  Logs:          $LOG_DIR/"
echo "=========================================="
echo ""
echo "Generated plots:"
ls -la "$RESULTS_DIR/plots/"*.png 2>/dev/null || echo "  (no plots found)"
echo ""
echo "Done! Cleaning up servers..."
