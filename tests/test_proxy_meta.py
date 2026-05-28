"""Tests for the proxy's per-request stamping: ``upstream_url`` on
captured requests and ``upstream_fingerprint`` on captured responses.

The proxy keeps no metadata file, no startup probe, no drift state —
those concerns moved to the export layer, derived from the per-row
stamps tested here.

Uses the same in-process ASGI wiring as ``test_proxy.py`` — proxy
talks to a Starlette mock upstream through an ``httpx.AsyncClient``
backed by ``ASGITransport``.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from agentcap.proxy import CHAT_COMPLETIONS_PATH, make_app


class UpstreamSpy:
    def __init__(self) -> None:
        self.responder = None

    def set_responder(self, fn) -> None:
        self.responder = fn


def _build_upstream(spy: UpstreamSpy) -> Starlette:
    async def chat_handler(request: Request) -> Response:
        if spy.responder is None:
            return JSONResponse({"error": "no responder"}, status_code=500)
        return await spy.responder(request)

    return Starlette(
        routes=[Route(CHAT_COMPLETIONS_PATH, chat_handler, methods=["POST"])]
    )


@pytest.fixture
def capture_dir(tmp_path: Path) -> Path:
    d = tmp_path / "capture"
    d.mkdir()
    return d


@pytest.fixture
def spy() -> UpstreamSpy:
    return UpstreamSpy()


@pytest.fixture
def proxy_app(spy: UpstreamSpy, capture_dir: Path):
    upstream_transport = httpx.ASGITransport(app=_build_upstream(spy))
    upstream_client = httpx.AsyncClient(
        transport=upstream_transport, base_url="http://upstream"
    )
    return make_app("http://upstream", capture_dir, client=upstream_client)


# ---------------------------------------------------------------------------
# Per-request: upstream_url stamping
# ---------------------------------------------------------------------------


def test_request_stamps_upstream_url(
    spy: UpstreamSpy, capture_dir: Path, proxy_app
):
    """Every ``.request.json`` carries the URL the proxy was forwarding
    to. Export derives the provider classification from this stamp
    alone — no sidecar metadata file involved."""
    async def responder(request):
        return JSONResponse({"id": "x", "model": "m", "choices": []})

    spy.set_responder(responder)
    with TestClient(proxy_app) as client:
        client.post(
            CHAT_COMPLETIONS_PATH,
            json={"model": "m", "messages": [{"role": "user", "content": "."}]},
        )

    req_files = list(capture_dir.glob("*.request.json"))
    assert len(req_files) == 1
    rec = json.loads(req_files[0].read_text())
    assert rec["upstream_url"] == "http://upstream"


def test_no_metadata_file_written(
    spy: UpstreamSpy, capture_dir: Path, proxy_app
):
    """The capture dir contains only per-request capture files — no
    ``_proxy.json``, no ``_meta.json``, nothing else."""
    async def responder(request):
        return JSONResponse({"id": "x", "model": "m", "choices": []})

    spy.set_responder(responder)
    with TestClient(proxy_app) as client:
        client.post(
            CHAT_COMPLETIONS_PATH,
            json={"model": "m", "messages": [{"role": "user", "content": "."}]},
        )

    names = sorted(p.name for p in capture_dir.iterdir())
    # One .request.json + one .response.json, nothing else.
    assert len(names) == 2
    assert all(
        n.endswith(".request.json") or n.endswith(".response.json")
        for n in names
    )


# ---------------------------------------------------------------------------
# Per-response: upstream_fingerprint stamping
# ---------------------------------------------------------------------------


def test_response_fingerprint_extracted_from_upstream_headers(
    spy: UpstreamSpy, capture_dir: Path, proxy_app
):
    async def responder(request):
        return JSONResponse(
            {
                "id": "x",
                "model": "qwen-actually-served",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}}],
            },
            headers={"server": "llama.cpp", "x-served-by": "fireworks-pod-7"},
        )

    spy.set_responder(responder)
    with TestClient(proxy_app) as client:
        resp = client.post(
            CHAT_COMPLETIONS_PATH,
            json={"model": "m", "messages": [{"role": "user", "content": "."}]},
        )
        assert resp.status_code == 200

    resp_files = list(capture_dir.glob("*.response.json"))
    assert len(resp_files) == 1
    rec = json.loads(resp_files[0].read_text())
    fp = rec["upstream_fingerprint"]
    assert fp["server"] == "llama.cpp"
    assert fp["x_served_by"] == "fireworks-pod-7"
    assert fp["served_model"] == "qwen-actually-served"
    assert fp["build_info"] is None  # not echoed on this response


def test_streaming_response_fingerprint_picks_model_from_first_chunk(
    spy: UpstreamSpy, capture_dir: Path, proxy_app
):
    """For SSE responses the body isn't a single dict; extract ``model``
    from the first parseable ``data:`` payload."""
    sse_chunks = [
        b'data: {"id":"x","model":"qwen-served","choices":[{"delta":{"role":"assistant"}}]}\n\n',
        b'data: {"id":"x","model":"qwen-served","choices":[{"delta":{"content":"hi"}}]}\n\n',
        b"data: [DONE]\n\n",
    ]

    async def responder(request):
        async def gen():
            for c in sse_chunks:
                yield c
        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={"server": "llama.cpp"},
        )

    spy.set_responder(responder)
    with TestClient(proxy_app) as client:
        with client.stream(
            "POST",
            CHAT_COMPLETIONS_PATH,
            json={"model": "m", "stream": True, "messages": [{"role": "user", "content": "."}]},
        ) as resp:
            for _ in resp.iter_bytes():
                pass

    rec = json.loads(next(capture_dir.glob("*.response.json")).read_text())
    assert rec["stream"] is True
    assert rec["upstream_fingerprint"]["served_model"] == "qwen-served"
    assert rec["upstream_fingerprint"]["server"] == "llama.cpp"


# ---------------------------------------------------------------------------
# Export-side provider derivation from the per-request stamp
# ---------------------------------------------------------------------------


def test_detect_provider_columns_derives_from_request_stamp(tmp_path: Path):
    from agentcap.export import detect_provider_columns

    capture = tmp_path / "t"
    capture.mkdir()
    (capture / "rid.request.json").write_text(json.dumps({
        "request_id": "rid",
        "captured_at": 1,
        "upstream_url": "https://router.huggingface.co",
        "body": {"model": "meta-llama/Llama-3.3-70B:fireworks-ai", "messages": []},
    }))
    cols = detect_provider_columns(capture)
    assert cols["upstream_url"] == "https://router.huggingface.co"
    assert cols["provider"] == "hf-router/fireworks-ai"


def test_detect_provider_columns_local_upstream(tmp_path: Path):
    from agentcap.export import detect_provider_columns

    capture = tmp_path / "t"
    capture.mkdir()
    (capture / "rid.request.json").write_text(json.dumps({
        "request_id": "rid",
        "captured_at": 1,
        "upstream_url": "http://127.0.0.1:8000",
        "body": {"model": "qwen-test", "messages": []},
    }))
    cols = detect_provider_columns(capture)
    assert cols["provider"] == "local"


def test_detect_provider_columns_empty_for_legacy_capture(tmp_path: Path):
    """Trace dir from before the proxy started stamping upstream_url —
    no way to derive the column, return empty so the parquet schema
    just omits it."""
    from agentcap.export import detect_provider_columns

    capture = tmp_path / "t"
    capture.mkdir()
    (capture / "rid.request.json").write_text(json.dumps({
        "request_id": "rid",
        "captured_at": 1,
        "body": {"model": "m", "messages": []},
    }))
    assert detect_provider_columns(capture) == {}


def test_detect_provider_columns_no_requests(tmp_path: Path):
    from agentcap.export import detect_provider_columns

    capture = tmp_path / "t"
    capture.mkdir()
    assert detect_provider_columns(capture) == {}
