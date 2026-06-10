"""
proxy_server.py
PD分离架构的Proxy路由服务器（对齐vLLM官方disagg_proxy_demo模式）。

核心机制：
- Proxy 接收请求后，先选择一个 decode 实例
- 将请求发给 prefill 实例（max_tokens=1），prefill 通过内部 NCCL 将 KV 传给 decode
- 再将原始请求发给 decode 实例完成生成
- KV cache 路由由 vLLM 内部的 kv_connector_extra_config 中的 proxy（端口30001）自动处理
"""

import argparse
import asyncio
import itertools
import json
import logging
import random
import time
from collections import defaultdict

import aiohttp
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="vLLM PD Disaggregated Proxy")

# --- Configuration (set via command-line args in main) ---
PREFILL_INSTANCES = ["localhost:8100"]
DECODE_INSTANCES = ["localhost:8201", "localhost:8202", "localhost:8203"]

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


async def forward_request(url: str, data: dict):
    """Forward a request to a vLLM instance and yield the response."""
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        headers = {"Authorization": "Bearer EMPTY"}
        async with session.post(url, json=data, headers=headers) as response:
            if response.status != 200:
                error_text = await response.text()
                raise HTTPException(status_code=response.status, detail=error_text)

            # Check if streaming
            content_type = response.headers.get("content-type", "")
            if "text/event-stream" in content_type:
                async for chunk in response.content.iter_any():
                    yield chunk
            else:
                body = await response.read()
                yield body


@app.post("/v1/completions")
async def proxy_completions(raw_request: Request):
    """
    Proxy /v1/completions with PD disaggregated routing.
    Follows the same pattern as vLLM's official disagg_proxy_demo.py.
    """
    request_body = await raw_request.json()
    bench_batch_id = request_body.pop("batch_id", "default")
    bench_request_id = request_body.pop("request_id", f"req-{time.time()}")
    original_max_tokens = request_body.get("max_tokens", 453)

    # Phase 1: Prefill (send to prefill instance with max_tokens=1)
    kv_prepare_request = request_body.copy()
    kv_prepare_request["max_tokens"] = 1

    prefill_instance = random.choice(PREFILL_INSTANCES)
    prefill_start = time.perf_counter()

    try:
        async for _ in forward_request(
            f"http://{prefill_instance}/v1/completions",
            kv_prepare_request,
        ):
            continue  # consume the prefill response (we don't need the output)
    except HTTPException as exc:
        raise exc

    prefill_end = time.perf_counter()
    prefill_time = prefill_end - prefill_start

    # Phase 2: Decode (send original request to a random decode instance)
    decode_instance = random.choice(DECODE_INSTANCES)
    decode_start = time.perf_counter()

    is_streaming = request_body.get("stream", False)

    if is_streaming:
        async def streaming_generator():
            async for chunk in forward_request(
                f"http://{decode_instance}/v1/completions",
                request_body,
            ):
                yield chunk

            # After streaming is done, record metrics
            decode_end = time.perf_counter()
            decode_time = decode_end - decode_start
            total_time = decode_end - prefill_start
            await record_metric(
                bench_batch_id, bench_request_id,
                prefill_time, decode_time, total_time, decode_instance,
            )

        return StreamingResponse(
            streaming_generator(),
            media_type="text/event-stream",
        )

    # Non-streaming
    response_body = b""
    async for chunk in forward_request(
        f"http://{decode_instance}/v1/completions",
        request_body,
    ):
        response_body += chunk

    decode_end = time.perf_counter()
    decode_time = decode_end - decode_start
    total_time = decode_end - prefill_start

    await record_metric(
        bench_batch_id, bench_request_id,
        prefill_time, decode_time, total_time, decode_instance,
    )

    # Parse and augment response with timing
    try:
        result = json.loads(response_body)
        result["timing"] = {
            "prefill_time_ms": round(prefill_time * 1000, 2),
            "decode_time_ms": round(decode_time * 1000, 2),
            "total_time_ms": round(total_time * 1000, 2),
            "decode_instance": decode_instance,
        }
        return JSONResponse(content=result)
    except json.JSONDecodeError:
        return JSONResponse(content={"raw": response_body.decode()})


@app.post("/v1/chat/completions")
async def proxy_chat_completions(raw_request: Request):
    """Proxy /v1/chat/completions with same PD routing logic."""
    request_body = await raw_request.json()
    bench_batch_id = request_body.pop("batch_id", "default")
    bench_request_id = request_body.pop("request_id", f"req-{time.time()}")
    original_max_tokens = request_body.get("max_tokens", 453)

    # Phase 1: Prefill
    kv_prepare_request = request_body.copy()
    kv_prepare_request["max_tokens"] = 1
    if "max_completion_tokens" in kv_prepare_request:
        kv_prepare_request["max_completion_tokens"] = 1

    prefill_instance = random.choice(PREFILL_INSTANCES)
    prefill_start = time.perf_counter()

    try:
        async for _ in forward_request(
            f"http://{prefill_instance}/v1/chat/completions",
            kv_prepare_request,
        ):
            continue
    except HTTPException as exc:
        raise exc

    prefill_end = time.perf_counter()
    prefill_time = prefill_end - prefill_start

    # Phase 2: Decode
    decode_instance = random.choice(DECODE_INSTANCES)
    decode_start = time.perf_counter()

    response_body = b""
    async for chunk in forward_request(
        f"http://{decode_instance}/v1/chat/completions",
        request_body,
    ):
        response_body += chunk

    decode_end = time.perf_counter()
    decode_time = decode_end - decode_start
    total_time = decode_end - prefill_start

    await record_metric(
        bench_batch_id, bench_request_id,
        prefill_time, decode_time, total_time, decode_instance,
    )

    try:
        result = json.loads(response_body)
        result["timing"] = {
            "prefill_time_ms": round(prefill_time * 1000, 2),
            "decode_time_ms": round(decode_time * 1000, 2),
            "total_time_ms": round(total_time * 1000, 2),
            "decode_instance": decode_instance,
        }
        return JSONResponse(content=result)
    except json.JSONDecodeError:
        return JSONResponse(content={"raw": response_body.decode()})


@app.get("/metrics")
async def get_metrics():
    """Return collected timing metrics."""
    async with metrics_lock:
        return JSONResponse(content=dict(metrics_store))


@app.post("/metrics/reset")
async def reset_metrics():
    """Clear all collected metrics."""
    async with metrics_lock:
        metrics_store.clear()
    return JSONResponse(content={"status": "metrics reset"})


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return JSONResponse(content={"status": "ok"})


def main():
    global PREFILL_INSTANCES, DECODE_INSTANCES

    parser = argparse.ArgumentParser(description="PD Disaggregated Proxy Server")
    parser.add_argument("--port", type=int, default=8000,
                        help="Proxy server port (default: 8000)")
    parser.add_argument("--prefill", type=str, nargs="+",
                        default=["localhost:8100"],
                        help="Prefill instance host:port list")
    parser.add_argument("--decode", type=str, nargs="+",
                        default=["localhost:8201", "localhost:8202", "localhost:8203"],
                        help="Decode instance host:port list")
    args = parser.parse_args()

    PREFILL_INSTANCES = args.prefill
    DECODE_INSTANCES = args.decode

    logger.info("=" * 50)
    logger.info(" PD Disaggregated Proxy Server")
    logger.info("=" * 50)
    logger.info(f"  Prefill instances: {PREFILL_INSTANCES}")
    logger.info(f"  Decode instances:  {DECODE_INSTANCES}")
    logger.info(f"  Proxy port:        {args.port}")
    logger.info("=" * 50)

    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
