"""
visualize_results.py
读取 benchmark 输出的 JSON 结果文件，生成可视化图表和 CSV 汇总表格。

图表包括：
1. Prefill 时间 vs Batch Size
2. Decode 时间 vs Batch Size
3. Prefill + Decode 合并对比图
4. 端到端延迟 vs Batch Size
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd


def load_results(filepath: str) -> dict:
    """Load benchmark results from JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def results_to_dataframe(data: dict) -> pd.DataFrame:
    """Convert results list to a pandas DataFrame."""
    rows = []
    for result in data["results"]:
        rows.append({
            "batch_size": result["batch_size"],
            "num_requests": result["num_requests"],
            "num_success": result["num_success"],
            "avg_prefill_ms": result["avg_prefill_ms"],
            "avg_decode_ms": result["avg_decode_ms"],
            "avg_total_ms": result["avg_total_ms"],
            "p50_prefill_ms": result["p50_prefill_ms"],
            "p50_decode_ms": result["p50_decode_ms"],
            "p99_prefill_ms": result["p99_prefill_ms"],
            "p99_decode_ms": result["p99_decode_ms"],
            "min_prefill_ms": result["min_prefill_ms"],
            "max_prefill_ms": result["max_prefill_ms"],
            "min_decode_ms": result["min_decode_ms"],
            "max_decode_ms": result["max_decode_ms"],
            "avg_output_tokens": result.get("avg_output_tokens", 0),
        })
    return pd.DataFrame(rows)


def plot_prefill_vs_batch_size(dataframe: pd.DataFrame, output_dir: str):
    """Plot Prefill time vs Batch Size."""
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(dataframe["batch_size"], dataframe["avg_prefill_ms"],
            marker="o", linewidth=2, label="Avg Prefill", color="#2196F3")
    ax.fill_between(
        dataframe["batch_size"],
        dataframe["min_prefill_ms"],
        dataframe["max_prefill_ms"],
        alpha=0.15, color="#2196F3", label="Min-Max Range",
    )
    ax.plot(dataframe["batch_size"], dataframe["p99_prefill_ms"],
            marker="s", linewidth=1, linestyle="--", label="P99 Prefill",
            color="#F44336", markersize=4)

    ax.set_xlabel("Batch Size", fontsize=12)
    ax.set_ylabel("Prefill Time (ms)", fontsize=12)
    ax.set_title("Prefill Time vs Batch Size\n(PD Disaggregated, Prefix Caching OFF)",
                 fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(dataframe["batch_size"])
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    filepath = os.path.join(output_dir, "prefill_vs_batch_size.png")
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"  Saved: {filepath}")


def plot_decode_vs_batch_size(dataframe: pd.DataFrame, output_dir: str):
    """Plot Decode time vs Batch Size."""
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(dataframe["batch_size"], dataframe["avg_decode_ms"],
            marker="o", linewidth=2, label="Avg Decode", color="#4CAF50")
    ax.fill_between(
        dataframe["batch_size"],
        dataframe["min_decode_ms"],
        dataframe["max_decode_ms"],
        alpha=0.15, color="#4CAF50", label="Min-Max Range",
    )
    ax.plot(dataframe["batch_size"], dataframe["p99_decode_ms"],
            marker="s", linewidth=1, linestyle="--", label="P99 Decode",
            color="#FF9800", markersize=4)

    ax.set_xlabel("Batch Size", fontsize=12)
    ax.set_ylabel("Decode Time (ms)", fontsize=12)
    ax.set_title("Decode Time vs Batch Size\n(PD Disaggregated, Prefix Caching OFF)",
                 fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(dataframe["batch_size"])
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    filepath = os.path.join(output_dir, "decode_vs_batch_size.png")
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"  Saved: {filepath}")


def plot_combined_comparison(dataframe: pd.DataFrame, output_dir: str):
    """Plot Prefill and Decode times together for comparison."""
    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(dataframe["batch_size"], dataframe["avg_prefill_ms"],
            marker="o", linewidth=2, label="Avg Prefill Time",
            color="#2196F3")
    ax.plot(dataframe["batch_size"], dataframe["avg_decode_ms"],
            marker="s", linewidth=2, label="Avg Decode Time",
            color="#4CAF50")
    ax.plot(dataframe["batch_size"], dataframe["p99_prefill_ms"],
            marker="^", linewidth=1, linestyle="--", label="P99 Prefill",
            color="#1565C0", markersize=4)
    ax.plot(dataframe["batch_size"], dataframe["p99_decode_ms"],
            marker="v", linewidth=1, linestyle="--", label="P99 Decode",
            color="#2E7D32", markersize=4)

    ax.set_xlabel("Batch Size", fontsize=12)
    ax.set_ylabel("Time (ms)", fontsize=12)
    ax.set_title(
        "Prefill vs Decode Time by Batch Size\n"
        "(PD Disaggregated 1P3D, Prefix Caching OFF)",
        fontsize=14,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(dataframe["batch_size"])
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    filepath = os.path.join(output_dir, "prefill_decode_comparison.png")
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"  Saved: {filepath}")


def plot_e2e_latency(dataframe: pd.DataFrame, output_dir: str):
    """Plot end-to-end latency vs Batch Size with stacked breakdown."""
    fig, ax = plt.subplots(figsize=(14, 7))

    ax.bar(dataframe["batch_size"], dataframe["avg_prefill_ms"],
           width=5, label="Prefill", color="#2196F3", alpha=0.85)
    ax.bar(dataframe["batch_size"], dataframe["avg_decode_ms"],
           width=5, bottom=dataframe["avg_prefill_ms"],
           label="Decode", color="#4CAF50", alpha=0.85)

    ax.plot(dataframe["batch_size"], dataframe["avg_total_ms"],
            marker="D", linewidth=2, label="Total E2E",
            color="#F44336", markersize=5)

    ax.set_xlabel("Batch Size", fontsize=12)
    ax.set_ylabel("Latency (ms)", fontsize=12)
    ax.set_title(
        "End-to-End Latency Breakdown by Batch Size\n"
        "(PD Disaggregated 1P3D, Prefix Caching OFF)",
        fontsize=14,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xticks(dataframe["batch_size"])
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    filepath = os.path.join(output_dir, "e2e_latency_breakdown.png")
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"  Saved: {filepath}")


def export_csv(dataframe: pd.DataFrame, output_dir: str):
    """Export summary table as CSV."""
    filepath = os.path.join(output_dir, "benchmark_summary.csv")
    dataframe.to_csv(filepath, index=False)
    print(f"  Saved: {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize vLLM PD benchmark results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i", type=str,
        default="results/benchmark_results.json",
        help="Input JSON results file",
    )
    parser.add_argument(
        "--output-dir", "-o", type=str,
        default="results/plots",
        help="Output directory for plots and CSV",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading results from: {args.input}")
    data = load_results(args.input)

    config = data.get("config", {})
    print(f"  Model:         {config.get('model', 'N/A')}")
    print(f"  Input tokens:  {config.get('input_tokens', 'N/A')}")
    print(f"  Output tokens: {config.get('output_tokens', 'N/A')}")
    print(f"  Timestamp:     {config.get('timestamp', 'N/A')}")

    dataframe = results_to_dataframe(data)
    print(f"  Data points:   {len(dataframe)}")
    print()

    print("Generating plots...")
    plot_prefill_vs_batch_size(dataframe, args.output_dir)
    plot_decode_vs_batch_size(dataframe, args.output_dir)
    plot_combined_comparison(dataframe, args.output_dir)
    plot_e2e_latency(dataframe, args.output_dir)
    print()

    print("Exporting CSV...")
    export_csv(dataframe, args.output_dir)
    print()

    # Print summary table to stdout
    print("=" * 110)
    print(" SUMMARY TABLE")
    print("=" * 110)
    summary_cols = [
        "batch_size", "num_success",
        "avg_prefill_ms", "p99_prefill_ms",
        "avg_decode_ms", "p99_decode_ms",
        "avg_total_ms",
    ]
    print(dataframe[summary_cols].to_string(index=False))
    print("=" * 110)
    print(f"\nAll outputs saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
