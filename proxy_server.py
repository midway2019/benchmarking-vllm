"""
proxy_server.py
PD分离架构的Proxy路由服务器。

功能：
1. 接收客户端请求
2. 转发到 Prefill 实例 (max_tokens=1) 完成 prefill
3. Prefill 完成后，将原始请求随机转发到 3 个 Decode 实例之一
4. 记录 prefill/decode 时间戳
5. 提供 /metrics 端点查询延迟数据
"""

import argparse
import asyncio
import json
import logging
import random
import time
from collections import defaultdict

import aiohttp
from quart import Quart, request, jsonify, Response

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = Quart(__name__)

# --- Configuration ---
PREFILL_URL = "http://localhost:8100"
DECODE_URLS = [
    "http://localhost:8201",
    "http://localhost:8202",
    "http://localhost:8203",
]
PROXY_PORT = 8000

# --- Metrics Storage ---
metrics_store = defaultdict(list)
metrics_lock = asyncio.Lock()


async def record_metric(batch_id: str, request_id: str, prefill_time: float,
                        decode_time: float, total_time: float,
                        decode_instance: str):
    """Record timing metrics for a single request."""
    async with metrics_lock:
        metrics_store[batch_id].append({
            "request_id": request_id,
            "prefill_time_ms": round(prefill_time * 1000, 2),
            "decode_time_ms": round(decode_time * 1000, 2),
            "total_time_ms": round(total_time * 1000, 2),
            "decode_instance": decode_instance,
            "timestamp": time.time(),
        })


@app.route("/v1/completions", methods=["POST"])
async def proxy_completions():
    """
    Proxy endpoint for /v1/completions.
    1. Send to prefill instance with max_tokens=1
    2. On completion, forward original request to a random decode instance
    3. Return decode result with timing metadata
    """
    original_body = await request.get_json()
    batch_id = original_body.pop("batch_id", "default")
    request_id = original_body.pop("request_id", f"req-{time.time()}")
    original_max_tokens = original_body.get("max_tokens", 453)

    is_streaming = original_body.get("stream", False)

    # --- Phase 1: Prefill ---
    prefill_body = {**original_body, "max_tokens": 1, "stream": False}
    prefill_start = time.perf_counter()

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=300)
    ) as session:
        async with session.post(
            f"{PREFILL_URL}/v1/completions",
            json=prefill_body,
            headers={"Content-Type": "application/json"},
        ) as prefill_resp:
            prefill_result = await prefill_resp.json()
            if prefill_resp.status != 200:
                return jsonify({
                    "error": "Prefill failed",
                    "detail": prefill_result,
                }), prefill_resp.status

    prefill_end = time.perf_counter()
    prefill_time = prefill_end - prefill_start

    # --- Phase 2: Decode (random instance) ---
    decode_url = random.choice(DECODE_URLS)
    decode_body = {**original_body, "max_tokens": original_max_tokens}

    decode_start = time.perf_counter()

    if is_streaming:
        # Streaming response: forward SSE stream from decode instance
        async def stream_generator():
            first_token_time = None
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=600)
            ) as session:
                async with session.post(
                    f"{decode_url}/v1/completions",
                    json=decode_body,
                    headers={"Content-Type": "application/json"},
                ) as decode_resp:
                    async for line in decode_resp.content:
                        decoded_line = line.decode("utf-8").strip()
                        if not decoded_line:
                            continue
                        if first_token_time is None and decoded_line.startswith("data:"):
                            first_token_time = time.perf_counter()
                        yield line

            decode_end = time.perf_counter()
            decode_time = decode_end - decode_start
            total_time = decode_end - prefill_start

            await record_metric(
                batch_id, request_id, prefill_time,
                decode_time, total_time, decode_url,
            )

            # Send timing metadata as final SSE event
            timing_data = json.dumps({
                "timing": {
                    "prefill_time_ms": round(prefill_time * 1000, 2),
                    "decode_time_ms": round(decode_time * 1000, 2),
                    "total_time_ms": round(total_time * 1000, 2),
                }
            })
            yield f"data: {timing_data}\n\n".encode()

        return Response(
            stream_generator(),
            content_type="text/event-stream",
        )

    # Non-streaming response
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=600)
    ) as session:
        async with session.post(
            f"{decode_url}/v1/completions",
            json=decode_body,
            headers={"Content-Type": "application/json"},
        ) as decode_resp:
            decode_result = await decode_resp.json()
            if decode_resp.status != 200:
                return jsonify({
                    "error": "Decode failed",
                    "detail": decode_result,
                }), decode_resp.status

    decode_end = time.perf_counter()
    decode_time = decode_end - decode_start
    total_time = decode_end - prefill_start

    await record_metric(
        batch_id, request_id, prefill_time,
        decode_time, total_time, decode_url,
    )

    decode_result["timing"] = {
        "prefill_time_ms": round(prefill_time * 1000, 2),
        "decode_time_ms": round(decode_time * 1000, 2),
        "total_time_ms": round(total_time * 1000, 2),
        "decode_instance": decode_url,
    }

    return jsonify(decode_result)


@app.route("/v1/chat/completions", methods=["POST"])
async def proxy_chat_completions():
    """Proxy endpoint for /v1/chat/completions with same PD routing logic."""
    original_body = await request.get_json()
    batch_id = original_body.pop("batch_id", "default")
    request_id = original_body.pop("request_id", f"req-{time.time()}")
    original_max_tokens = original_body.get("max_tokens", 453)

    # --- Phase 1: Prefill ---
    prefill_body = {**original_body, "max_tokens": 1, "stream": False}
    prefill_start = time.perf_counter()

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=300)
    ) as session:
        async with session.post(
            f"{PREFILL_URL}/v1/chat/completions",
            json=prefill_body,
            headers={"Content-Type": "application/json"},
        ) as prefill_resp:
            prefill_result = await prefill_resp.json()
            if prefill_resp.status != 200:
                return jsonify({
                    "error": "Prefill failed",
                    "detail": prefill_result,
                }), prefill_resp.status

    prefill_end = time.perf_counter()
    prefill_time = prefill_end - prefill_start

    # --- Phase 2: Decode (random instance) ---
    decode_url = random.choice(DECODE_URLS)
    decode_body = {**original_body, "max_tokens": original_max_tokens, "stream": False}

    decode_start = time.perf_counter()

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=600)
    ) as session:
        async with session.post(
            f"{decode_url}/v1/chat/completions",
            json=decode_body,
            headers={"Content-Type": "application/json"},
        ) as decode_resp:
            decode_result = await decode_resp.json()
            if decode_resp.status != 200:
                return jsonify({
                    "error": "Decode failed",
                    "detail": decode_result,
                }), decode_resp.status

    decode_end = time.perf_counter()
    decode_time = decode_end - decode_start
    total_time = decode_end - prefill_start

    await record_metric(
        batch_id, request_id, prefill_time,
        decode_time, total_time, decode_url,
    )

    decode_result["timing"] = {
        "prefill_time_ms": round(prefill_time * 1000, 2),
        "decode_time_ms": round(decode_time * 1000, 2),
        "total_time_ms": round(total_time * 1000, 2),
        "decode_instance": decode_url,
    }

    return jsonify(decode_result)


@app.route("/metrics", methods=["GET"])
async def get_metrics():
    """Return collected timing metrics."""
    async with metrics_lock:
        return jsonify(dict(metrics_store))


@app.route("/metrics/reset", methods=["POST"])
async def reset_metrics():
    """Clear all collected metrics."""
    async with metrics_lock:
        metrics_store.clear()
    return jsonify({"status": "metrics reset"})


@app.route("/health", methods=["GET"])
async def health_check():
    """Health check endpoint."""
    return jsonify({"status": "ok"})


def parse_args():
    parser = argparse.ArgumentParser(description="PD Disaggregated Proxy Server")
    parser.add_argument("--port", type=int, default=PROXY_PORT,
                        help="Proxy server port (default: 8000)")
    parser.add_argument("--prefill-url", type=str, default=PREFILL_URL,
                        help="Prefill instance URL")
    parser.add_argument("--decode-urls", type=str, nargs="+",
                        default=DECODE_URLS,
                        help="Decode instance URLs (space-separated)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    PREFILL_URL = args.prefill_url
    DECODE_URLS = args.decode_urls
    app.run(host="0.0.0.0", port=args.port)
