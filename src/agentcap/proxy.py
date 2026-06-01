"""Capture proxy for OpenAI-compat chat completions.

Captures ``POST /v1/chat/completions`` to
``<capture_dir>/<request_id>.{request,response}.json``; other paths
pass through. Streaming responses are forwarded chunk-by-chunk and
the assembled bytes persisted at end-of-stream.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Optional

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from starlette.routing import Route


# Constant so per-agent Containerfiles can bake the proxy URL into
# the agent's config files without per-run rewriting.
IN_PROCESS_PROXY_HOST = "127.0.0.1"
IN_PROCESS_PROXY_PORT = 0  # kernel-assigned ephemeral; read back via ProxyHandle.port

CHAT_COMPLETIONS_PATH = "/v1/chat/completions"

# Hop-by-hop (RFC 7230 §6.1) plus content-length / content-encoding
# which the framework recomputes from the re-emitted body.
_HOP_BY_HOP = frozenset({
    "host",
    "content-length",
    "content-encoding",
    "transfer-encoding",
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "upgrade",
})


def _filter_headers(headers: Any) -> dict[str, str]:
    return {k: v for k, v in headers.items() if k.lower() not in _HOP_BY_HOP}


def _safe_json_loads(raw: bytes) -> Any:
    """Parse JSON; on failure, return a {"raw": <decoded>} placeholder so
    the capture stays well-formed even on malformed input."""
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return {"_unparsed_raw": raw.decode("utf-8", errors="replace")}


def _lower_headers(headers: Any) -> dict[str, str]:
    try:
        return {k.lower(): v for k, v in headers.items()}
    except AttributeError:
        return {}


def _extract_model_from_sse(raw: bytes) -> str | None:
    """Find a ``"model"`` field in the first parseable SSE data line."""
    for line in raw.splitlines():
        if not line.startswith(b"data:"):
            continue
        payload = line[len(b"data:"):].strip()
        if not payload or payload == b"[DONE]":
            continue
        try:
            obj = json.loads(payload)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict):
            m = obj.get("model")
            if isinstance(m, str) and m:
                return m
    return None


def _response_fingerprint(headers: Any, body_obj: Any) -> dict[str, str | None]:
    h = _lower_headers(headers)
    served_model: str | None = None
    if isinstance(body_obj, dict):
        m = body_obj.get("model")
        if isinstance(m, str) and m:
            served_model = m
    return {
        "server": h.get("server") or None,
        "x_served_by": h.get("x-served-by") or None,
        "via": h.get("via") or None,
        "build_info": h.get("x-build-info") or None,
        "served_model": served_model,
    }


class CaptureProxy:
    """Capture proxy as a Starlette handler bundle.

    Pass a custom ``client`` (typically ``httpx.AsyncClient`` with
    ``ASGITransport``) to wire against a mock upstream in tests.
    """

    def __init__(
        self,
        upstream: str,
        capture_dir: Path | str,
        *,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self.upstream = upstream.rstrip("/")
        self.capture_dir = Path(capture_dir)
        self.capture_dir.mkdir(parents=True, exist_ok=True)
        self._client = client
        self._owns_client = client is None
        # Context the orchestrator sets before each turn — stamped into
        # each captured request so rid → (task_id, turn) is recoverable
        # from the capture file alone, no sidecar mapping.
        self._task_id: str | None = None
        self._turn: int | None = None

    def set_context(self, *, task_id: str | None, turn: int | None) -> None:
        self._task_id = task_id
        self._turn = turn

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            # No timeout: agent calls can be long, agent decides when to give up.
            self._client = httpx.AsyncClient(timeout=None)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()

    def _persist_request(self, request_id: str, body_bytes: bytes, captured_at: int) -> None:
        path = self.capture_dir / f"{request_id}.request.json"
        record = {
            "request_id": request_id,
            "captured_at": captured_at,
            "upstream_url": self.upstream,
            "task_id": self._task_id,
            "turn": self._turn,
            "body": _safe_json_loads(body_bytes),
        }
        path.write_text(json.dumps(record, indent=2))

    def _persist_response_nonstream(
        self,
        request_id: str,
        status_code: int,
        body_bytes: bytes,
        captured_at: int,
        upstream_headers: Any,
    ) -> None:
        body = _safe_json_loads(body_bytes)
        fp = _response_fingerprint(upstream_headers, body)
        path = self.capture_dir / f"{request_id}.response.json"
        record = {
            "request_id": request_id,
            "captured_at_resp": captured_at,
            "stream": False,
            "status_code": status_code,
            "body": body,
            "upstream_fingerprint": fp,
        }
        path.write_text(json.dumps(record, indent=2))

    def _persist_response_stream(
        self,
        request_id: str,
        status_code: int,
        raw_bytes: bytes,
        captured_at: int,
        upstream_headers: Any,
    ) -> None:
        sse_model = _extract_model_from_sse(raw_bytes)
        synthetic_body = {"model": sse_model} if sse_model else None
        fp = _response_fingerprint(upstream_headers, synthetic_body)
        path = self.capture_dir / f"{request_id}.response.json"
        record = {
            "request_id": request_id,
            "captured_at_resp": captured_at,
            "stream": True,
            "status_code": status_code,
            "raw": raw_bytes.decode("utf-8", errors="replace"),
            "upstream_fingerprint": fp,
        }
        path.write_text(json.dumps(record, indent=2))

    async def chat_completions(self, request: Request) -> Response:
        body_bytes = await request.body()
        body_obj = _safe_json_loads(body_bytes)
        is_stream = bool(isinstance(body_obj, dict) and body_obj.get("stream", False))

        request_id = uuid.uuid4().hex
        self._persist_request(request_id, body_bytes, int(time.time()))

        url = f"{self.upstream}{CHAT_COMPLETIONS_PATH}"
        fwd_headers = _filter_headers(request.headers)
        client = await self._get_client()

        if is_stream:
            return await self._forward_stream(
                client, url, body_bytes, fwd_headers, request_id
            )
        return await self._forward_nonstream(
            client, url, body_bytes, fwd_headers, request_id
        )

    async def _forward_nonstream(
        self,
        client: httpx.AsyncClient,
        url: str,
        body_bytes: bytes,
        fwd_headers: dict[str, str],
        request_id: str,
    ) -> Response:
        upstream_resp = await client.post(url, content=body_bytes, headers=fwd_headers)
        resp_bytes = upstream_resp.content
        self._persist_response_nonstream(
            request_id,
            upstream_resp.status_code,
            resp_bytes,
            int(time.time()),
            upstream_resp.headers,
        )
        return Response(
            content=resp_bytes,
            status_code=upstream_resp.status_code,
            headers=_filter_headers(upstream_resp.headers),
            media_type=upstream_resp.headers.get("content-type"),
        )

    async def _forward_stream(
        self,
        client: httpx.AsyncClient,
        url: str,
        body_bytes: bytes,
        fwd_headers: dict[str, str],
        request_id: str,
    ) -> StreamingResponse:
        # We need the upstream status + content-type before we can
        # construct the StreamingResponse. Open the stream eagerly,
        # capture metadata, then yield bytes lazily.
        async def streamer() -> AsyncIterator[bytes]:
            chunks: list[bytes] = []
            status_code = 502
            upstream_headers: Any = {}
            try:
                async with client.stream(
                    "POST", url, content=body_bytes, headers=fwd_headers
                ) as upstream_resp:
                    status_code = upstream_resp.status_code
                    upstream_headers = upstream_resp.headers
                    async for chunk in upstream_resp.aiter_bytes():
                        chunks.append(chunk)
                        yield chunk
            finally:
                self._persist_response_stream(
                    request_id,
                    status_code,
                    b"".join(chunks),
                    int(time.time()),
                    upstream_headers,
                )

        return StreamingResponse(streamer(), media_type="text/event-stream")

    async def passthrough(self, request: Request) -> Response:
        url = f"{self.upstream}{request.url.path}"
        if request.url.query:
            url = f"{url}?{request.url.query}"
        body_bytes = await request.body()
        fwd_headers = _filter_headers(request.headers)
        client = await self._get_client()
        upstream_resp = await client.request(
            request.method,
            url,
            content=body_bytes if body_bytes else None,
            headers=fwd_headers,
        )
        return Response(
            content=upstream_resp.content,
            status_code=upstream_resp.status_code,
            headers=_filter_headers(upstream_resp.headers),
            media_type=upstream_resp.headers.get("content-type"),
        )


def make_app(
    upstream: str,
    capture_dir: Path | str,
    *,
    client: Optional[httpx.AsyncClient] = None,
) -> Starlette:
    """Build the Starlette ASGI app wrapping a CaptureProxy."""
    proxy = CaptureProxy(upstream, capture_dir, client=client)
    routes = [
        Route(CHAT_COMPLETIONS_PATH, proxy.chat_completions, methods=["POST"]),
        Route(
            "/{full_path:path}",
            proxy.passthrough,
            methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"],
        ),
    ]

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app: Starlette):
        try:
            yield
        finally:
            await proxy.aclose()

    app = Starlette(routes=routes, lifespan=lifespan)
    app.state.proxy = proxy
    return app


def serve(
    upstream: str,
    capture_dir: Path | str,
    host: str = "127.0.0.1",
    port: int = 8001,
) -> None:
    import uvicorn

    app = make_app(upstream, capture_dir)
    uvicorn.run(app, host=host, port=port)


class ProxyHandle:
    """Running in-process proxy. Use as a context manager."""

    def __init__(
        self, server, thread, host: str, port: int,
        proxy: CaptureProxy | None = None,
    ) -> None:
        self._server = server
        self._thread = thread
        self.host = host
        self.port = port
        self.proxy = proxy

    def set_context(self, *, task_id: str | None, turn: int | None) -> None:
        """Forward to the underlying ``CaptureProxy`` so subsequent
        captures are stamped with the given orchestrator-turn context."""
        if self.proxy is not None:
            self.proxy.set_context(task_id=task_id, turn=turn)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def shutdown(self, *, timeout: float = 10) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=timeout)

    def __enter__(self) -> "ProxyHandle":
        return self

    def __exit__(self, *exc) -> None:
        self.shutdown()


def serve_in_thread(
    upstream: str,
    capture_dir: Path | str,
    host: str = IN_PROCESS_PROXY_HOST,
    port: int = IN_PROCESS_PROXY_PORT,
    *,
    log_level: str = "warning",
    startup_timeout: float = 10.0,
) -> ProxyHandle:
    """Start the proxy on a daemon thread; block until uvicorn is bound.

    With ``port=0`` the kernel-assigned port is read back into
    ``ProxyHandle.port``.
    """
    import threading
    import time

    import uvicorn

    app = make_app(upstream, capture_dir)
    config = uvicorn.Config(app, host=host, port=port, log_level=log_level)
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.time() + startup_timeout
    while not server.started:
        if time.time() > deadline:
            server.should_exit = True
            thread.join(timeout=2)
            raise RuntimeError(
                f"proxy did not start within {startup_timeout}s on {host}:{port}"
            )
        time.sleep(0.05)

    bound_host, bound_port = host, port
    try:
        bound_host, bound_port = server.servers[0].sockets[0].getsockname()[:2]
    except (AttributeError, IndexError, TypeError):
        pass

    return ProxyHandle(server, thread, bound_host, bound_port, proxy=app.state.proxy)
