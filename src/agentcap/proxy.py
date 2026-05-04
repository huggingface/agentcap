"""Transparent OpenAI-compat HTTP capture proxy.

Sits between an agent CLI and a model server. For every POST to
``/v1/chat/completions``, persists the raw request body and the
response body to ``<trace_dir>/<request_id>.{request,response}.json``.
Other paths (e.g. ``/v1/models``) pass through transparently with no
capture.

Streaming responses are forwarded chunk-by-chunk to the client and the
assembled raw bytes are persisted at end-of-stream.

The capture layer is intentionally "dumb": no tokenisation, no
chat-template render, no per-token metadata. Manifest computation is
the export layer's job, run offline against a captured trace dir.
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


CHAT_COMPLETIONS_PATH = "/v1/chat/completions"

# Hop-by-hop headers we never forward. RFC 7230 §6.1 plus a couple of
# pragmatic additions (content-length / content-encoding will be
# recomputed by the framework and would clash with our re-emitted body).
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
    the trace stays well-formed even on malformed input."""
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return {"_unparsed_raw": raw.decode("utf-8", errors="replace")}


class CaptureProxy:
    """The capture proxy as a Starlette-compatible handler bundle.

    ``upstream`` is the base URL of the model server — no path prefix,
    e.g. ``http://127.0.0.1:8000``. The proxy mirrors the incoming
    request path verbatim onto ``upstream``.

    Pass a custom ``client`` (typically an ``httpx.AsyncClient`` with an
    ``ASGITransport``) to wire the proxy against a mock upstream in
    tests.
    """

    def __init__(
        self,
        upstream: str,
        trace_dir: Path | str,
        *,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self.upstream = upstream.rstrip("/")
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self._client = client
        self._owns_client = client is None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            # No timeout: agent calls can be long. The agent is in
            # control of when to give up; we just relay.
            self._client = httpx.AsyncClient(timeout=None)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()

    def _persist_request(self, request_id: str, body_bytes: bytes, captured_at: int) -> None:
        path = self.trace_dir / f"{request_id}.request.json"
        record = {
            "request_id": request_id,
            "captured_at": captured_at,
            "body": _safe_json_loads(body_bytes),
        }
        path.write_text(json.dumps(record, indent=2))

    def _persist_response_nonstream(
        self,
        request_id: str,
        status_code: int,
        body_bytes: bytes,
        captured_at: int,
    ) -> None:
        path = self.trace_dir / f"{request_id}.response.json"
        record = {
            "request_id": request_id,
            "captured_at_resp": captured_at,
            "stream": False,
            "status_code": status_code,
            "body": _safe_json_loads(body_bytes),
        }
        path.write_text(json.dumps(record, indent=2))

    def _persist_response_stream(
        self,
        request_id: str,
        status_code: int,
        raw_bytes: bytes,
        captured_at: int,
    ) -> None:
        path = self.trace_dir / f"{request_id}.response.json"
        record = {
            "request_id": request_id,
            "captured_at_resp": captured_at,
            "stream": True,
            "status_code": status_code,
            "raw": raw_bytes.decode("utf-8", errors="replace"),
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
            try:
                async with client.stream(
                    "POST", url, content=body_bytes, headers=fwd_headers
                ) as upstream_resp:
                    status_code = upstream_resp.status_code
                    async for chunk in upstream_resp.aiter_bytes():
                        chunks.append(chunk)
                        yield chunk
            finally:
                self._persist_response_stream(
                    request_id,
                    status_code,
                    b"".join(chunks),
                    int(time.time()),
                )

        # Probe upstream for headers first by issuing a HEAD-like
        # round-trip is expensive; SSE handlers usually emit
        # ``text/event-stream``. Default to that and let the upstream
        # override via the wrapping app's machinery.
        return StreamingResponse(streamer(), media_type="text/event-stream")

    async def passthrough(self, request: Request) -> Response:
        """Transparent forward for any path other than chat completions.
        No capture."""
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
    trace_dir: Path | str,
    *,
    client: Optional[httpx.AsyncClient] = None,
) -> Starlette:
    """Build the Starlette ASGI app wrapping a CaptureProxy."""
    proxy = CaptureProxy(upstream, trace_dir, client=client)
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
    # Stash the proxy so callers (esp. tests) can reach it for
    # introspection without poking through Starlette internals.
    app.state.proxy = proxy
    return app


def serve(
    upstream: str,
    trace_dir: Path | str,
    host: str = "127.0.0.1",
    port: int = 8001,
) -> None:
    """Run the proxy under uvicorn. Production entrypoint."""
    import uvicorn

    app = make_app(upstream, trace_dir)
    uvicorn.run(app, host=host, port=port)


class ProxyHandle:
    """A running in-process proxy. Use as a context manager."""

    def __init__(self, server, thread, host: str, port: int) -> None:
        self._server = server
        self._thread = thread
        self.host = host
        self.port = port

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
    trace_dir: Path | str,
    host: str = "127.0.0.1",
    port: int = 8001,
    *,
    log_level: str = "warning",
    startup_timeout: float = 10.0,
) -> ProxyHandle:
    """Start the proxy on a daemon thread and return a handle.

    Blocks until the underlying uvicorn server reports ``started``, so
    callers can immediately point an agent at the returned ``base_url``.
    """
    import threading
    import time

    import uvicorn

    app = make_app(upstream, trace_dir)
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

    return ProxyHandle(server, thread, host, port)
