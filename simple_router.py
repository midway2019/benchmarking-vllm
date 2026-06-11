#!/usr/bin/env python3
"""
simple_router.py
极简 PD 分离路由器，替代 sglang_router。

工作原理：
  1. 收到请求后，先转发到 prefill 实例
  2. SGLang 的 PD 分离机制会自动将 KV cache 传输到 decode 实例
  3. decode 实例生成 token 并返回给客户端

本路由器只需将请求发到 prefill 实例，SGLang 内部处理 prefill→decode 的调度。
"""

import argparse
import asyncio
import itertools
import logging
import time
from aiohttp import web, ClientSession, ClientTimeout

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


class SimpleRouter:
    def __init__(self, prefill_urls: list, decode_urls: list):
        self.prefill_urls = prefill_urls
        self.decode_urls = decode_urls
        self.prefill_cycle = itertools.cycle(prefill_urls)
        self.decode_cycle = itertools.cycle(decode_urls)
        self.request_count = 0
        self.session = None

    async def init_session(self):
        timeout = ClientTimeout(total=600)
        self.session = ClientSession(timeout=timeout)

    async def close_session(self):
        if self.session:
            await self.session.close()

    async def health(self, request):
        return web.json_response({
            "status": "ok",
            "prefill_instances": self.prefill_urls,
            "decode_instances": self.decode_urls,
            "request_count": self.request_count,
        })

    async def proxy_request(self, request):
        """Forward request to prefill instance. SGLang handles PD routing internally."""
        self.request_count += 1
        target_url = next(self.prefill_cycle)
        path = request.path

        try:
            body = await request.read()
            headers = dict(request.headers)
            headers.pop("Host", None)
            headers.pop("host", None)

            target = f"{target_url}{path}"
            logger.info(f"[{self.request_count}] {request.method} {path} -> {target}")

            async with self.session.request(
                method=request.method,
                url=target,
                data=body,
                headers=headers,
            ) as resp:
                # Check if streaming response
                if resp.headers.get("Transfer-Encoding") == "chunked" or \
                   "text/event-stream" in resp.headers.get("Content-Type", ""):
                    response = web.StreamResponse(
                        status=resp.status,
                        headers={
                            "Content-Type": resp.headers.get("Content-Type", "text/event-stream"),
                            "Cache-Control": "no-cache",
                        },
                    )
                    await response.prepare(request)
                    async for chunk in resp.content.iter_any():
                        await response.write(chunk)
                    await response.write_eof()
                    return response
                else:
                    resp_body = await resp.read()
                    return web.Response(
                        status=resp.status,
                        body=resp_body,
                        content_type=resp.headers.get("Content-Type", "application/json"),
                    )

        except Exception as e:
            logger.error(f"Proxy error: {e}")
            return web.json_response(
                {"error": str(e)},
                status=502,
            )

    async def get_models(self, request):
        """Forward GET /v1/models to prefill instance."""
        return await self.proxy_request(request)


async def create_app(args):
    router_obj = SimpleRouter(
        prefill_urls=args.prefill,
        decode_urls=args.decode,
    )
    await router_obj.init_session()

    app = web.Application()
    app.router.add_get("/health", router_obj.health)
    app.router.add_route("*", "/v1/{tail:.*}", router_obj.proxy_request)
    app.router.add_route("*", "/generate", router_obj.proxy_request)

    async def on_cleanup(app):
        await router_obj.close_session()

    app.on_cleanup.append(on_cleanup)

    logger.info(f"Simple PD Router starting on port {args.port}")
    logger.info(f"  Prefill: {args.prefill}")
    logger.info(f"  Decode:  {args.decode}")

    return app


def main():
    parser = argparse.ArgumentParser(description="Simple PD Router")
    parser.add_argument("--prefill", nargs="+", required=True,
                        help="Prefill instance URLs")
    parser.add_argument("--decode", nargs="+", required=True,
                        help="Decode instance URLs")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=29001)
    args = parser.parse_args()

    web.run_app(create_app(args), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
