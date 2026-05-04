"""Integration tests for the capture proxy — real HTTP over TCP loopback.

Two uvicorn servers run in worker threads:
  - mock upstream (Starlette app on a free port)
  - capture proxy (Starlette app on another free port, pointed at upstream)

The test client makes real ``httpx.Client`` HTTP calls to the proxy.
This catches wiring issues that the in-process ASGITransport unit
tests in ``test_proxy.py`` would not — header reconstruction, content
encoding, streaming-chunk pump-through, etc.

Marked as ``integration`` so they can be filtered out with
``pytest -m 'not integration'`` when iterating on logic.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from pathlib import Path

import httpx
import pytest
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from agentcap.proxy import CHAT_COMPLETIONS_PATH, make_app


pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class UvicornThreadServer:
    """Run a uvicorn server in a daemon thread and shut it down cleanly."""

    def __init__(self, app, host: str = "127.0.0.1", port: int | None = None):
        self.host = host
        self.port = port or _free_port()
        config = uvicorn.Config(
            app,
            host=host,
            port=self.port,
            log_level="error",
            lifespan="on",
        )
        self.server = uvicorn.Server(config)
        # Disable signal handler installation — we're not the main thread.
        self.server.install_signal_handlers = lambda *_: None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self, timeout: float = 5.0) -> None:
        self._thread = threading.Thread(target=self.server.run, daemon=True)
        self._thread.start()
        deadline = time.monotonic() + timeout
        # Poll until the server is accepting connections — server.started
        # flips to True once uvicorn's serve() has bound the socket.
        while time.monotonic() < deadline:
            if self.server.started:
                # Smoke-check the socket is actually accepting.
                try:
                    with socket.create_connection((self.host, self.port), timeout=0.2):
                        return
                except OSError:
                    pass
            time.sleep(0.05)
        raise RuntimeError(f"uvicorn server on :{self.port} failed to start in {timeout}s")

    def stop(self, timeout: float = 5.0) -> None:
        self.server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                # Best-effort: force-exit. uvicorn's force_exit triggers
                # the loop to exit immediately on the next iteration.
                self.server.force_exit = True
                self._thread.join(timeout=timeout)


# ---------------------------------------------------------------------------
# Mock upstream
# ---------------------------------------------------------------------------


class UpstreamSpy:
    def __init__(self) -> None:
        self.received_bodies: list[dict] = []
        self.received_paths: list[str] = []
        self.responder = None

    def set_responder(self, fn) -> None:
        self.responder = fn


def _build_upstream(spy: UpstreamSpy) -> Starlette:
    async def chat_handler(request: Request) -> Response:
        body = await request.body()
        try:
            spy.received_bodies.append(json.loads(body))
        except json.JSONDecodeError:
            spy.received_bodies.append({"_raw": body.decode("utf-8", errors="replace")})
        spy.received_paths.append(request.url.path)
        if spy.responder is None:
            return JSONResponse({"error": "no responder"}, status_code=500)
        return await spy.responder(request)

    async def models_handler(request: Request) -> Response:
        spy.received_paths.append(request.url.path)
        return JSONResponse(
            {"object": "list", "data": [{"id": "real-mock", "object": "model"}]}
        )

    return Starlette(
        routes=[
            Route(CHAT_COMPLETIONS_PATH, chat_handler, methods=["POST"]),
            Route("/v1/models", models_handler, methods=["GET"]),
        ]
    )


# ---------------------------------------------------------------------------
# Fixtures — two real uvicorn servers + a clean trace dir per test
# ---------------------------------------------------------------------------


@pytest.fixture
def spy() -> UpstreamSpy:
    return UpstreamSpy()


@pytest.fixture
def trace_dir(tmp_path: Path) -> Path:
    d = tmp_path / "trace"
    d.mkdir()
    return d


@pytest.fixture
def upstream(spy: UpstreamSpy):
    server = UvicornThreadServer(_build_upstream(spy))
    server.start()
    yield server
    server.stop()


@pytest.fixture
def proxy(upstream: UvicornThreadServer, trace_dir: Path):
    # Real proxy → real upstream URL. No client injection: the proxy
    # creates its own httpx.AsyncClient and dials over TCP loopback.
    proxy_app = make_app(upstream.url, trace_dir)
    server = UvicornThreadServer(proxy_app)
    server.start()
    yield server
    server.stop()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_chat_nonstreaming_over_real_http(
    spy: UpstreamSpy, trace_dir: Path, proxy: UvicornThreadServer
):
    async def responder(request):
        return JSONResponse(
            {
                "id": "chatcmpl-http-1",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "ok"}}
                ],
            }
        )

    spy.set_responder(responder)
    body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "ping"}],
        "stream": False,
    }
    with httpx.Client(timeout=10.0) as client:
        resp = client.post(f"{proxy.url}{CHAT_COMPLETIONS_PATH}", json=body)
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "ok"

    # Upstream saw the body verbatim
    assert spy.received_bodies == [body]

    # Trace dir got both files
    req_files = list(trace_dir.glob("*.request.json"))
    resp_files = list(trace_dir.glob("*.response.json"))
    assert len(req_files) == 1 and len(resp_files) == 1
    assert json.loads(req_files[0].read_text())["body"] == body
    assert (
        json.loads(resp_files[0].read_text())["body"]["choices"][0]["message"]["content"]
        == "ok"
    )


def test_chat_streaming_over_real_http(
    spy: UpstreamSpy, trace_dir: Path, proxy: UvicornThreadServer
):
    sse_chunks = [
        b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":" world"}}]}\n\n',
        b"data: [DONE]\n\n",
    ]

    async def responder(request):
        async def gen():
            for c in sse_chunks:
                yield c

        return StreamingResponse(gen(), media_type="text/event-stream")

    spy.set_responder(responder)
    body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "stream me"}],
        "stream": True,
    }
    received = bytearray()
    with httpx.Client(timeout=10.0) as client:
        with client.stream(
            "POST", f"{proxy.url}{CHAT_COMPLETIONS_PATH}", json=body
        ) as resp:
            assert resp.status_code == 200
            for chunk in resp.iter_bytes():
                received.extend(chunk)

    expected = b"".join(sse_chunks)
    assert bytes(received) == expected

    resp_files = list(trace_dir.glob("*.response.json"))
    assert len(resp_files) == 1
    record = json.loads(resp_files[0].read_text())
    assert record["stream"] is True
    assert record["status_code"] == 200
    assert record["raw"] == expected.decode("utf-8")


def test_passthrough_over_real_http_does_not_capture(
    spy: UpstreamSpy, trace_dir: Path, proxy: UvicornThreadServer
):
    with httpx.Client(timeout=10.0) as client:
        resp = client.get(f"{proxy.url}/v1/models")
    assert resp.status_code == 200
    assert resp.json() == {
        "object": "list",
        "data": [{"id": "real-mock", "object": "model"}],
    }
    # Upstream got the call, trace dir untouched
    assert "/v1/models" in spy.received_paths
    assert list(trace_dir.iterdir()) == []
