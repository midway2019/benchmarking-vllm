"""
proxy_server.py
PD分离架构的Proxy路由服务器。

基于 vLLM 官方 disagg_proxy_p2p_nccl_xpyd.py 的核心机制：
1. 端口30001: ZMQ服务发现 - vLLM实例启动时通过ZMQ注册自己
2. 端口10001: HTTP代理 - 接收客户端请求，构造带地址的request_id
3. request_id格式: ___prefill_addr_{zmq_addr}___decode_addr_{zmq_addr}_{uuid}
4. 通过 X-Request-Id header 传递给 vLLM 实例

附加功能：记录 prefill/decode 时间用于 benchmark。
"""

import argparse
import os
import socket
import threading
import time
import uuid
from collections import defaultdict
from typing import Any

import aiohttp
import msgpack
import zmq
from quart import Quart, make_response, request, jsonify

# --- Service Discovery State ---
prefill_instances: dict[str, Any] = {}  # http_address: (zmq_address, stamp)
decode_instances: dict[str, Any] = {}   # http_address: (zmq_address, stamp)
prefill_cv = threading.Condition()
decode_cv = threading.Condition()

DEFAULT_PING_SECONDS = 15
count = 0

# --- Metrics Storage ---
metrics_store: dict[str, list] = defaultdict(list)
import asyncio
metrics_lock = asyncio.Lock()

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)
app = Quart(__name__)


def random_uuid() -> str:
    return str(uuid.uuid4().hex)


# ========================
# ZMQ Service Discovery
# ========================

def _remove_oldest_instances(instances: dict[str, Any]) -> None:
    oldest_key = next(iter(instances), None)
    while oldest_key is not None:
        value = instances[oldest_key]
        if value[1] > time.time():
            break
        print(f"Remove [HTTP:{oldest_key}, ZMQ:{value[0]}, stamp:{value[1]}]")
        instances.pop(oldest_key, None)
        oldest_key = next(iter(instances), None)


def _listen_for_register(poller, router_socket):
    global prefill_instances, decode_instances
    global prefill_cv, decode_cv
    while True:
        socks = dict(poller.poll())
        if router_socket in socks:
            remote_address, message = router_socket.recv_multipart()
            data = msgpack.loads(message)
            if data["type"] == "P":
                with prefill_cv:
                    node = prefill_instances.get(data["http_address"], None)
                    prefill_instances[data["http_address"]] = (
                        data["zmq_address"],
                        time.time() + DEFAULT_PING_SECONDS,
                    )
                    _remove_oldest_instances(prefill_instances)
            elif data["type"] == "D":
                with decode_cv:
                    node = decode_instances.get(data["http_address"], None)
                    decode_instances[data["http_address"]] = (
                        data["zmq_address"],
                        time.time() + DEFAULT_PING_SECONDS,
                    )
                    _remove_oldest_instances(decode_instances)
            else:
                print(f"Unexpected message from {remote_address}, data: {data}")
                continue
            if node is None:
                print(f"Add [HTTP:{data['http_address']}, ZMQ:{data['zmq_address']}]")


def start_service_discovery(hostname, port):
    if not hostname:
        hostname = socket.gethostname()
    if port == 0:
        raise ValueError("Port cannot be 0")

    context = zmq.Context()
    router_socket = context.socket(zmq.ROUTER)
    router_socket.bind(f"tcp://{hostname}:{port}")

    poller = zmq.Poller()
    poller.register(router_socket, zmq.POLLIN)

    listener_thread = threading.Thread(
        target=_listen_for_register,
        args=[poller, router_socket],
        daemon=True,
    )
    listener_thread.start()
    return listener_thread


# ========================
# HTTP Request Forwarding
# ========================

async def forward_request(url, data, request_id):
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', 'EMPTY')}",
            "X-Request-Id": request_id,
        }
        async with session.post(url=url, json=data, headers=headers) as response:
            if response.status == 200:
                async for chunk_bytes in response.content.iter_chunked(1024):
                    yield chunk_bytes


# ========================
# HTTP Proxy Endpoints
# ========================

@app.route("/v1/completions", methods=["POST"])
@app.route("/v1/chat/completions", methods=["POST"])
async def handle_request():
    try:
        original_request_data = await request.get_json()

        # Extract benchmark metadata (won't be forwarded to vLLM)
        bench_batch_id = original_request_data.pop("batch_id", "default")
        bench_request_id = original_request_data.pop("request_id", f"req-{time.time()}")

        prefill_request = original_request_data.copy()
        prefill_request["max_tokens"] = 1
        if "max_completion_tokens" in prefill_request:
            prefill_request["max_completion_tokens"] = 1

        global count, prefill_instances, decode_instances
        global prefill_cv, decode_cv

        # Wait for at least one prefill and one decode instance
        retry = 0
        while retry < 60:
            with prefill_cv:
                prefill_list = list(prefill_instances.items())
            with decode_cv:
                decode_list = list(decode_instances.items())
            if prefill_list and decode_list:
                break
            print(f"Waiting for instances... prefill={len(prefill_list)}, decode={len(decode_list)}")
            await asyncio.sleep(1)
            retry += 1

        if not prefill_list or not decode_list:
            return await make_response(
                ({"error": "No prefill/decode instances registered"}, 503)
            )

        # Round-robin selection
        prefill_addr, prefill_zmq_addr = prefill_list[count % len(prefill_list)]
        prefill_zmq_addr = prefill_zmq_addr[0]

        decode_addr, decode_zmq_addr = decode_list[count % len(decode_list)]
        decode_zmq_addr = decode_zmq_addr[0]

        print(
            f"handle_request count:{count}, "
            f"[HTTP:{prefill_addr}, ZMQ:{prefill_zmq_addr}] -> "
            f"[HTTP:{decode_addr}, ZMQ:{decode_zmq_addr}]"
        )
        count += 1

        # Build request_id with embedded addresses (critical for P2pNcclConnector)
        request_id = (
            f"___prefill_addr_{prefill_zmq_addr}___"
            f"decode_addr_{decode_zmq_addr}_{random_uuid()}"
        )

        # Phase 1: Prefill
        prefill_start = time.perf_counter()
        async for _ in forward_request(
            f"http://{prefill_addr}{request.path}",
            prefill_request,
            request_id,
        ):
            continue
        prefill_end = time.perf_counter()
        prefill_time = prefill_end - prefill_start

        # Phase 2: Decode
        decode_start = time.perf_counter()
        generator = forward_request(
            f"http://{decode_addr}{request.path}",
            original_request_data,
            request_id,
        )

        # Wrap generator to record timing after completion
        async def timed_generator():
            async for chunk in generator:
                yield chunk
            decode_end = time.perf_counter()
            decode_time = decode_end - decode_start
            total_time = decode_end - prefill_start
            async with metrics_lock:
                metrics_store[bench_batch_id].append({
                    "request_id": bench_request_id,
                    "prefill_time_ms": round(prefill_time * 1000, 2),
                    "decode_time_ms": round(decode_time * 1000, 2),
                    "total_time_ms": round(total_time * 1000, 2),
                    "decode_instance": decode_addr,
                    "timestamp": time.time(),
                })

        response = await make_response(timed_generator())
        response.timeout = None
        return response

    except Exception as exc:
        import sys
        import traceback
        exc_info = sys.exc_info()
        print("Error occurred in disagg prefill proxy server")
        print(exc)
        print("".join(traceback.format_exception(*exc_info)))
        return await make_response(({"error": str(exc)}, 500))


@app.route("/metrics", methods=["GET"])
async def get_metrics():
    async with metrics_lock:
        return await make_response(jsonify(dict(metrics_store)))


@app.route("/metrics/reset", methods=["POST"])
async def reset_metrics():
    async with metrics_lock:
        metrics_store.clear()
    return await make_response(jsonify({"status": "metrics reset"}))


@app.route("/health", methods=["GET"])
async def health_check():
    return await make_response(jsonify({
        "status": "ok",
        "prefill_instances": len(prefill_instances),
        "decode_instances": len(decode_instances),
    }))


def parse_args():
    parser = argparse.ArgumentParser(description="PD Disaggregated Proxy Server")
    parser.add_argument("--zmq-port", type=int, default=39001,
                        help="ZMQ service discovery port (default: 39001)")
    parser.add_argument("--http-port", type=int, default=29001,
                        help="HTTP proxy port (default: 29001)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("=" * 50)
    print(" PD Disaggregated Proxy Server")
    print("=" * 50)
    print(f"  ZMQ discovery port: {args.zmq_port}")
    print(f"  HTTP proxy port:    {args.http_port}")
    print("=" * 50)

    t = start_service_discovery("0.0.0.0", args.zmq_port)
    app.run(host="0.0.0.0", port=args.http_port)
    t.join()
