# SGLang PD Disaggregated Benchmark

基于 SGLang 的 Disaggregated Prefill/Decode (PD分离) 架构 benchmark 工具，使用 NIXL 传输后端，精确测量不同 batch size 下的 prefill 时间和 decode 时间。

## 架构概览

```
                    ┌─────────────────┐
                    │  Benchmark Client│
                    │ (benchmark_pd.py)│
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  sglang_router   │
                    │  (port 29001)    │
                    └───┬─────────┬───┘
                        │         │
              Phase 1   │         │  Phase 2
              Prefill   │         │  Decode
                        │         │
               ┌────────▼──┐  ┌──▼───────────────────────┐
               │  Prefill   │  │   Decode Instances       │
               │  GPU 0     │  │   GPU 1 (port 29201)     │
               │  port 29100│  │   GPU 2 (port 29202)     │
               │            │  │   GPU 3 (port 29203)     │
               └────────────┘  └──────────────────────────┘
                    ◄──── NIXL KV Transfer (NVLink) ────►
```

## 测试参数

| 参数 | 默认值 |
|------|--------|
| 模型 | `meta-llama/Meta-Llama-3.1-8B-Instruct` |
| Input token 长度 | 453 |
| Output token 长度 | 453 |
| Batch sizes | 16, 32, 48, ..., 384, 392 |
| PD 比例 | 1:3 (1 Prefill + 3 Decode) |
| 传输后端 | NIXL (GPU-to-GPU via NVLink) |
| Radix Cache | 关闭 |

## 环境要求

- Python 3.10+
- CUDA 12.8
- 4 张 GPU（NVLink 互联，如 A100/H100 SXM）
- SGLang >= 0.5.0
- sglang-router >= 0.1.0
- nixl[cu12]

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
pip install "nixl[cu12]"
```

### 2. 一键运行（推荐）

```bash
bash run_benchmark.sh
```

### 3. 分步运行

```bash
# Step 1: 启动 PD 集群 + sglang_router
bash launch_pd_cluster.sh &

# Step 2: 等待所有实例就绪后，运行 benchmark
python benchmark_pd.py \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --input-len 453 \
    --output-len 453 \
    --proxy-url http://localhost:29001 \
    --output results/benchmark_results.json

# Step 3: 生成可视化
python visualize_results.py \
    --input results/benchmark_results.json \
    --output-dir results/plots
```

## 环境变量配置

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `HF_MODEL_NAME` | `meta-llama/Meta-Llama-3.1-8B-Instruct` | HuggingFace 模型名称 |
| `MAX_MODEL_LEN` | `1024` | 最大模型长度 |
| `MEM_FRACTION` | `0.85` | GPU 显存利用率 |
| `INPUT_LEN` | `453` | 输入 token 长度 |
| `OUTPUT_LEN` | `453` | 输出 token 长度 |
| `ROUTER_PORT` | `29001` | sglang_router 端口 |

## 输出说明

### 结果文件

```
results/
├── benchmark_results.json    # 完整 benchmark 结果（JSON）
└── plots/
    ├── benchmark_summary.csv          # 汇总表格
    ├── prefill_vs_batch_size.png      # Prefill 时间图
    ├── decode_vs_batch_size.png       # Decode 时间图
    ├── prefill_decode_comparison.png  # 对比图
    └── e2e_latency_breakdown.png     # 端到端延迟分解图
```

### 测量指标

- **Prefill Time (TTFT)**: 从请求发送到收到第一个 token 的时间
- **Decode Time**: 从第一个 token 到最后一个 token 的时间
- **Total Time (E2E)**: 端到端总延迟
- 每个 batch size 报告：avg / p50 / p99 / min / max

## 项目结构

```
benchmarking_vllm/
├── README.md                 # 本文件
├── requirements.txt          # Python 依赖
├── launch_pd_cluster.sh      # SGLang PD 集群 + router 启动脚本
├── benchmark_pd.py           # 核心 benchmark 程序
├── visualize_results.py      # 可视化与报告生成
├── run_benchmark.sh          # 一键运行脚本
├── logs/                     # 运行日志
└── results/                  # 测试结果与图表
```

## 注意事项

1. **Radix Cache 已关闭**：所有 SGLang 实例均使用 `--disable-radix-cache` 参数启动，确保每次 prefill 都是完整计算
2. **NIXL 传输后端**：使用 NVIDIA 官方 NIXL 库进行 GPU-to-GPU KV cache 传输，适合 NVLink 互联的单机多卡环境
3. **GPU 拓扑**：确保 4 张 GPU 之间有 NVLink 互联，可通过 `nvidia-smi topo -m` 检查
4. **显存**：建议每张 GPU 至少 24GB 显存（如 A100/H100 SXM）
