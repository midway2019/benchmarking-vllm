"""
benchmark_pd.py
SGLang PD分离架构 Benchmark 核心脚本。

在固定 input/output token 长度下，遍历不同 batch size，
通过 streaming API 精确测量每个请求的 prefill 时间 (TTFT) 和 decode 时间。

请求流程：
  客户端 → sglang_router → Prefill 实例 → Decode 实例 → 客户端

测量方式：
  - Prefill 时间 = TTFT (Time To First Token)
  - Decode 时间 = 最后一个 token 时间 - 第一个 token 时间
"""

import argparse
import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

import aiohttp
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class RequestResult:
    """Timing result for a single request."""
    request_id: str
    batch_size: int
    send_time: float = 0.0
    first_token_time: float = 0.0
    end_time: float = 0.0
    prefill_time_ms: float = 0.0
    decode_time_ms: float = 0.0
    total_time_ms: float = 0.0
    output_tokens: int = 0
    success: bool = True
    error: Optional[str] = None


@dataclass
class BatchResult:
    """Aggregated result for a batch size."""
    batch_size: int
    num_requests: int = 0
    num_success: int = 0
    avg_prefill_ms: float = 0.0
    avg_decode_ms: float = 0.0
    avg_total_ms: float = 0.0
    p50_prefill_ms: float = 0.0
    p50_decode_ms: float = 0.0
    p99_prefill_ms: float = 0.0
    p99_decode_ms: float = 0.0
    min_prefill_ms: float = 0.0
    max_prefill_ms: float = 0.0
    min_decode_ms: float = 0.0
    max_decode_ms: float = 0.0
    avg_output_tokens: float = 0.0
    requests: list = field(default_factory=list)


def generate_fixed_length_prompt(tokenizer, target_token_count: int) -> str:
    """
    Generate a prompt with exactly target_token_count tokens.
    Uses a repeated pattern and trims/pads to exact length.
    """
    base_text = (
        "The quick brown fox jumps over the lazy dog. "
        "In a world where technology advances rapidly, "
        "artificial intelligence continues to transform industries. "
        "Machine learning models are becoming increasingly powerful. "
    )

    # Repeat to get enough tokens
    repeated_text = (base_text + " ") * (target_token_count // 10 + 10)
    tokens = tokenizer.encode(repeated_text)

    # Trim to exact target length
    if len(tokens) >= target_token_count:
        tokens = tokens[:target_token_count]
    else:
        # Pad with repeated tokens if needed
        while len(tokens) < target_token_count:
            tokens.extend(tokenizer.encode(base_text))
        tokens = tokens[:target_token_count]

    prompt = tokenizer.decode(tokens, skip_special_tokens=True)

    # Verify token count
    actual_count = len(tokenizer.encode(prompt))
    if actual_count != target_token_count:
        # Fine-tune by adding/removing words
        while len(tokenizer.encode(prompt)) > target_token_count:
            prompt = " ".join(prompt.split()[:-1])
        while len(tokenizer.encode(prompt)) < target_token_count:
            prompt += " hello"
        # Final trim at token level
        tokens = tokenizer.encode(prompt)[:target_token_count]
        prompt = tokenizer.decode(tokens, skip_special_tokens=True)

    return prompt


async def send_single_request(
    session: aiohttp.ClientSession,
    base_url: str,
    prompt: str,
    max_tokens: int,
    model: str,
    request_id: str,
    batch_size: int,
    use_chat_api: bool = False,
) -> RequestResult:
    """Send a single streaming request and measure TTFT and decode time."""
    result = RequestResult(request_id=request_id, batch_size=batch_size)

    if use_chat_api:
        endpoint = f"{base_url}/v1/chat/completions"
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": True,
            "temperature": 0.0,
        }
    else:
        endpoint = f"{base_url}/v1/completions"
        body = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "stream": True,
            "temperature": 0.0,
        }

    try:
        result.send_time = time.perf_counter()

        async with session.post(
            endpoint,
            json=body,
            headers={"Content-Type": "application/json"},
        ) as resp:
            if resp.status != 200:
                error_body = await resp.text()
                result.success = False
                result.error = f"HTTP {resp.status}: {error_body[:200]}"
                return result

            token_count = 0
            async for raw_line in resp.content:
                line = raw_line.decode("utf-8").strip()
                if not line or not line.startswith("data:"):
                    continue

                data_str = line[len("data:"):].strip()
                if data_str == "[DONE]":
                    break

                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                # Check if this is a timing metadata event from proxy
                if "timing" in data and "choices" not in data:
                    continue

                # Count tokens from choices
                choices = data.get("choices", [])
                if not choices:
                    continue

                choice = choices[0]
                # For completions API
                text = choice.get("text", "")
                # For chat completions API
                delta = choice.get("delta", {})
                content = delta.get("content", "")

                if text or content:
                    token_count += 1
                    if token_count == 1:
                        result.first_token_time = time.perf_counter()

            result.end_time = time.perf_counter()
            result.output_tokens = token_count

            if result.first_token_time > 0:
                result.prefill_time_ms = (
                    (result.first_token_time - result.send_time) * 1000
                )
                result.decode_time_ms = (
                    (result.end_time - result.first_token_time) * 1000
                )
            result.total_time_ms = (result.end_time - result.send_time) * 1000

    except asyncio.TimeoutError:
        result.success = False
        result.error = "Request timed out"
    except Exception as exc:
        result.success = False
        result.error = str(exc)

    return result


def compute_batch_stats(results: list[RequestResult], batch_size: int) -> BatchResult:
    """Compute aggregate statistics for a batch of request results."""
    successful = [r for r in results if r.success]

    batch_result = BatchResult(
        batch_size=batch_size,
        num_requests=len(results),
        num_success=len(successful),
    )

    if not successful:
        return batch_result

    prefill_times = [r.prefill_time_ms for r in successful]
    decode_times = [r.decode_time_ms for r in successful]
    total_times = [r.total_time_ms for r in successful]
    output_tokens_list = [r.output_tokens for r in successful]

    batch_result.avg_prefill_ms = round(float(np.mean(prefill_times)), 2)
    batch_result.avg_decode_ms = round(float(np.mean(decode_times)), 2)
    batch_result.avg_total_ms = round(float(np.mean(total_times)), 2)

    batch_result.p50_prefill_ms = round(float(np.percentile(prefill_times, 50)), 2)
    batch_result.p50_decode_ms = round(float(np.percentile(decode_times, 50)), 2)
    batch_result.p99_prefill_ms = round(float(np.percentile(prefill_times, 99)), 2)
    batch_result.p99_decode_ms = round(float(np.percentile(decode_times, 99)), 2)

    batch_result.min_prefill_ms = round(float(np.min(prefill_times)), 2)
    batch_result.max_prefill_ms = round(float(np.max(prefill_times)), 2)
    batch_result.min_decode_ms = round(float(np.min(decode_times)), 2)
    batch_result.max_decode_ms = round(float(np.max(decode_times)), 2)

    batch_result.avg_output_tokens = round(float(np.mean(output_tokens_list)), 1)

    batch_result.requests = [asdict(r) for r in successful]

    return batch_result


async def run_batch_benchmark(
    base_url: str,
    prompt: str,
    max_tokens: int,
    model: str,
    batch_size: int,
    use_chat_api: bool = False,
    request_timeout: int = 600,
) -> BatchResult:
    """Run benchmark for a single batch size: send batch_size concurrent requests."""

    connector = aiohttp.TCPConnector(limit=batch_size + 10)
    timeout = aiohttp.ClientTimeout(total=request_timeout)

    async with aiohttp.ClientSession(
        connector=connector, timeout=timeout,
    ) as session:
        tasks = []
        for idx in range(batch_size):
            request_id = f"bs{batch_size}-req{idx}"
            task = send_single_request(
                session=session,
                base_url=base_url,
                prompt=prompt,
                max_tokens=max_tokens,
                model=model,
                request_id=request_id,
                batch_size=batch_size,
                use_chat_api=use_chat_api,
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks)

    return compute_batch_stats(list(results), batch_size)


async def warmup(
    base_url: str,
    prompt: str,
    max_tokens: int,
    model: str,
    num_warmup: int = 3,
    use_chat_api: bool = False,
):
    """Send a few warmup requests to prime the system."""
    logger.info(f"Running {num_warmup} warmup requests...")

    connector = aiohttp.TCPConnector(limit=10)
    timeout = aiohttp.ClientTimeout(total=300)

    async with aiohttp.ClientSession(
        connector=connector, timeout=timeout,
    ) as session:
        tasks = []
        for idx in range(num_warmup):
            task = send_single_request(
                session=session,
                base_url=base_url,
                prompt=prompt,
                max_tokens=min(max_tokens, 32),
                model=model,
                request_id=f"warmup-{idx}",
                batch_size=0,
                use_chat_api=use_chat_api,
            )
            tasks.append(task)

        warmup_results = await asyncio.gather(*tasks)

    successful = sum(1 for r in warmup_results if r.success)
    logger.info(f"Warmup complete: {successful}/{num_warmup} succeeded")

    if successful == 0:
        raise RuntimeError("All warmup requests failed! Check server connectivity.")


async def main():
    args = parse_args()

    # Generate batch sizes: [16, 32, 48, ..., 384, 392]
    batch_sizes = list(range(16, 385, 16))  # [16, 32, ..., 384]
    if 392 not in batch_sizes:
        batch_sizes.append(392)

    logger.info("=" * 60)
    logger.info(" SGLang PD Disaggregated Benchmark")
    logger.info("=" * 60)
    logger.info(f"  Model:          {args.model}")
    logger.info(f"  Input tokens:   {args.input_len}")
    logger.info(f"  Output tokens:  {args.output_len}")
    logger.info(f"  Batch sizes:    {batch_sizes}")
    logger.info(f"  Proxy URL:      {args.proxy_url}")
    logger.info(f"  Use Chat API:   {args.use_chat_api}")
    logger.info(f"  Num warmup:     {args.num_warmup}")
    logger.info("=" * 60)

    # Load tokenizer and generate fixed-length prompt
    logger.info(f"Loading tokenizer for {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    logger.info(f"Generating prompt with exactly {args.input_len} tokens...")
    prompt = generate_fixed_length_prompt(tokenizer, args.input_len)
    actual_tokens = len(tokenizer.encode(prompt))
    logger.info(f"  Generated prompt: {actual_tokens} tokens, {len(prompt)} chars")

    # Warmup
    await warmup(
        base_url=args.proxy_url,
        prompt=prompt,
        max_tokens=args.output_len,
        model=args.model,
        num_warmup=args.num_warmup,
        use_chat_api=args.use_chat_api,
    )

    # Run benchmark for each batch size
    all_results = []
    logger.info("")
    logger.info("Starting benchmark...")

    for batch_size in tqdm(batch_sizes, desc="Batch sizes"):
        logger.info(f"\n--- Batch Size: {batch_size} ---")

        batch_result = await run_batch_benchmark(
            base_url=args.proxy_url,
            prompt=prompt,
            max_tokens=args.output_len,
            model=args.model,
            batch_size=batch_size,
            use_chat_api=args.use_chat_api,
            request_timeout=args.timeout,
        )

        all_results.append(batch_result)

        logger.info(
            f"  BS={batch_size:>3d} | "
            f"Success: {batch_result.num_success}/{batch_result.num_requests} | "
            f"Prefill: {batch_result.avg_prefill_ms:>8.1f}ms (p99: {batch_result.p99_prefill_ms:>8.1f}ms) | "
            f"Decode:  {batch_result.avg_decode_ms:>8.1f}ms (p99: {batch_result.p99_decode_ms:>8.1f}ms) | "
            f"Total:   {batch_result.avg_total_ms:>8.1f}ms"
        )

        # Brief pause between batch sizes to let the system stabilize
        if batch_size < batch_sizes[-1]:
            await asyncio.sleep(args.inter_batch_delay)

    # Save results
    output_data = {
        "config": {
            "model": args.model,
            "input_tokens": args.input_len,
            "output_tokens": args.output_len,
            "batch_sizes": batch_sizes,
            "proxy_url": args.proxy_url,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "results": [],
    }

    for batch_result in all_results:
        result_dict = asdict(batch_result)
        # Remove per-request details from summary output to keep file manageable
        if not args.include_details:
            result_dict.pop("requests", None)
        output_data["results"].append(result_dict)

    output_file = args.output
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"\nResults saved to {output_file}")

    # Print summary table
    print_summary_table(all_results)


def print_summary_table(results: list[BatchResult]):
    """Print a formatted summary table to stdout."""
    print("\n" + "=" * 100)
    print(" BENCHMARK RESULTS SUMMARY")
    print("=" * 100)
    print(
        f"{'Batch':>6s} | {'Success':>8s} | "
        f"{'Prefill(avg)':>12s} | {'Prefill(p99)':>12s} | "
        f"{'Decode(avg)':>12s} | {'Decode(p99)':>12s} | "
        f"{'Total(avg)':>12s}"
    )
    print("-" * 100)
    for result in results:
        print(
            f"{result.batch_size:>6d} | "
            f"{result.num_success:>3d}/{result.num_requests:<3d} | "
            f"{result.avg_prefill_ms:>10.1f}ms | "
            f"{result.p99_prefill_ms:>10.1f}ms | "
            f"{result.avg_decode_ms:>10.1f}ms | "
            f"{result.p99_decode_ms:>10.1f}ms | "
            f"{result.avg_total_ms:>10.1f}ms"
        )
    print("=" * 100)


def parse_args():
    parser = argparse.ArgumentParser(
        description="SGLang PD Disaggregated Benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", type=str,
        default="LLM-Research/Meta-Llama-3.1-8B-Instruct",
        help="Model name for tokenizer and API requests",
    )
    parser.add_argument(
        "--input-len", type=int, default=453,
        help="Fixed input token length",
    )
    parser.add_argument(
        "--output-len", type=int, default=453,
        help="Fixed output token length (max_tokens)",
    )
    parser.add_argument(
        "--proxy-url", type=str, default="http://localhost:29001",
        help="SGLang router URL",
    )
    parser.add_argument(
        "--output", "-o", type=str, default="results/benchmark_results.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--num-warmup", type=int, default=3,
        help="Number of warmup requests",
    )
    parser.add_argument(
        "--timeout", type=int, default=600,
        help="Request timeout in seconds",
    )
    parser.add_argument(
        "--inter-batch-delay", type=float, default=5.0,
        help="Delay (seconds) between batch sizes for system stabilization",
    )
    parser.add_argument(
        "--use-chat-api", action="store_true",
        help="Use /v1/chat/completions instead of /v1/completions",
    )
    parser.add_argument(
        "--include-details", action="store_true",
        help="Include per-request details in output JSON",
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main())
