"""Tests for the capture proxy.

Strategy: stand up a mock upstream Starlette app and wire the proxy's
internal httpx client to it via ``ASGITransport``. Then drive the
proxy via Starlette's ``TestClient`` and assert on (a) what bytes the
agent-side client sees, and (b) what files land on disk in the trace
dir.

End-to-end network sockets are not used — everything runs in-process.
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


# ---------------------------------------------------------------------------
# Mock upstream — a tiny Starlette app that pretends to be an OpenAI-compat
# model server. Each test parameterises its behaviour by setting attributes
# on the wrapping ``Holder``.
# ---------------------------------------------------------------------------


class UpstreamSpy:
    """Records what the proxy forwarded to upstream + lets each test
    plug in a custom response factory."""

    def __init__(self) -> None:
        self.received_bodies: list[dict] = []
        self.received_headers: list[dict] = []
        self.received_paths: list[str] = []
        self.responder = None  # async callable: (request) -> Response

    def set_responder(self, fn) -> None:
        self.responder = fn


def _build_upstream(spy: UpstreamSpy) -> Starlette:
    async def chat_handler(request: Request) -> Response:
        body = await request.body()
        try:
            spy.received_bodies.append(json.loads(body))
        except json.JSONDecodeError:
            spy.received_bodies.append({"_unparsed": body.decode("utf-8", errors="replace")})
        spy.received_headers.append(dict(request.headers))
        spy.received_paths.append(request.url.path)
        if spy.responder is None:
            return JSONResponse({"error": "no responder configured"}, status_code=500)
        return await spy.responder(request)

    async def models_handler(request: Request) -> Response:
        spy.received_paths.append(request.url.path)
        return JSONResponse(
            {"object": "list", "data": [{"id": "mock-model", "object": "model"}]}
        )

    async def echo_handler(request: Request) -> Response:
        spy.received_paths.append(request.url.path)
        return JSONResponse({"path": request.url.path, "method": request.method})

    return Starlette(
        routes=[
            Route(CHAT_COMPLETIONS_PATH, chat_handler, methods=["POST"]),
            Route("/v1/models", models_handler, methods=["GET"]),
            Route(
                "/{anything:path}",
                echo_handler,
                methods=["GET", "POST", "PUT", "DELETE"],
            ),
        ]
    )


@pytest.fixture
def spy() -> UpstreamSpy:
    return UpstreamSpy()


@pytest.fixture
def trace_dir(tmp_path: Path) -> Path:
    d = tmp_path / "trace"
    d.mkdir()
    return d


@pytest.fixture
def proxy_client(spy: UpstreamSpy, trace_dir: Path):
    """A TestClient hitting the proxy, where the proxy talks to the
    mock upstream via ASGITransport."""
    upstream_app = _build_upstream(spy)
    upstream_transport = httpx.ASGITransport(app=upstream_app)
    upstream_client = httpx.AsyncClient(
        transport=upstream_transport, base_url="http://upstream"
    )
    proxy_app = make_app("http://upstream", trace_dir, client=upstream_client)
    with TestClient(proxy_app) as client:
        yield client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_chat_nonstreaming_captures_request_and_response(
    spy: UpstreamSpy, trace_dir: Path, proxy_client: TestClient
):
    async def responder(request):
        return JSONResponse(
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hi back"},
                        "finish_reason": "stop",
                    }
                ],
            }
        )

    spy.set_responder(responder)
    body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
    }
    resp = proxy_client.post(CHAT_COMPLETIONS_PATH, json=body)
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "hi back"

    # Upstream saw the same body
    assert spy.received_bodies == [body]
    assert spy.received_paths == [CHAT_COMPLETIONS_PATH]

    # Trace dir has exactly one request + response pair
    req_files = sorted(trace_dir.glob("*.request.json"))
    resp_files = sorted(trace_dir.glob("*.response.json"))
    assert len(req_files) == 1
    assert len(resp_files) == 1
    assert req_files[0].stem.split(".")[0] == resp_files[0].stem.split(".")[0]

    req_record = json.loads(req_files[0].read_text())
    assert req_record["body"] == body
    assert "request_id" in req_record
    assert isinstance(req_record["captured_at"], int)

    resp_record = json.loads(resp_files[0].read_text())
    assert resp_record["stream"] is False
    assert resp_record["status_code"] == 200
    assert resp_record["body"]["choices"][0]["message"]["content"] == "hi back"


def test_chat_streaming_forwards_chunks_and_captures_raw(
    spy: UpstreamSpy, trace_dir: Path, proxy_client: TestClient
):
    sse_chunks = [
        b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":" back"}}]}\n\n',
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
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }
    with proxy_client.stream("POST", CHAT_COMPLETIONS_PATH, json=body) as resp:
        assert resp.status_code == 200
        received = b"".join(resp.iter_bytes())

    # The agent-side client got the bytes the upstream produced
    assert received == b"".join(sse_chunks)

    # The trace's response.json captured the assembled stream + status
    resp_files = list(trace_dir.glob("*.response.json"))
    assert len(resp_files) == 1
    record = json.loads(resp_files[0].read_text())
    assert record["stream"] is True
    assert record["status_code"] == 200
    assert record["raw"] == b"".join(sse_chunks).decode("utf-8")


def test_passthrough_models_endpoint_is_not_captured(
    spy: UpstreamSpy, trace_dir: Path, proxy_client: TestClient
):
    resp = proxy_client.get("/v1/models")
    assert resp.status_code == 200
    assert resp.json() == {
        "object": "list",
        "data": [{"id": "mock-model", "object": "model"}],
    }
    # Upstream saw the call
    assert "/v1/models" in spy.received_paths
    # But nothing was written to the trace dir
    assert list(trace_dir.iterdir()) == []


def test_arbitrary_passthrough_path(
    spy: UpstreamSpy, trace_dir: Path, proxy_client: TestClient
):
    resp = proxy_client.get("/unrelated/path?x=1")
    assert resp.status_code == 200
    body = resp.json()
    assert body["path"] == "/unrelated/path"
    assert body["method"] == "GET"
    # Trace dir untouched
    assert list(trace_dir.iterdir()) == []


def test_two_requests_get_distinct_request_ids(
    spy: UpstreamSpy, trace_dir: Path, proxy_client: TestClient
):
    async def responder(request):
        return JSONResponse({"id": "x", "choices": []})

    spy.set_responder(responder)

    body = {"model": "m", "messages": [{"role": "user", "content": "."}]}
    proxy_client.post(CHAT_COMPLETIONS_PATH, json=body)
    proxy_client.post(CHAT_COMPLETIONS_PATH, json=body)

    req_files = sorted(trace_dir.glob("*.request.json"))
    assert len(req_files) == 2
    ids = {json.loads(p.read_text())["request_id"] for p in req_files}
    assert len(ids) == 2  # distinct


def test_malformed_request_body_still_captured(
    spy: UpstreamSpy, trace_dir: Path, proxy_client: TestClient
):
    async def responder(request):
        return JSONResponse({"choices": []})

    spy.set_responder(responder)

    raw = b"{not json"
    resp = proxy_client.post(
        CHAT_COMPLETIONS_PATH, content=raw, headers={"content-type": "application/json"}
    )
    # Upstream still got the bytes verbatim — we don't sanitise input.
    # Whether upstream accepts it is upstream's problem; we just relay.
    assert resp.status_code == 200

    req_files = list(trace_dir.glob("*.request.json"))
    assert len(req_files) == 1
    record = json.loads(req_files[0].read_text())
    # Body is preserved as a placeholder dict instead of crashing
    assert record["body"] == {"_unparsed_raw": "{not json"}


def test_upstream_500_is_forwarded_and_captured(
    spy: UpstreamSpy, trace_dir: Path, proxy_client: TestClient
):
    async def responder(request):
        return JSONResponse({"error": {"message": "boom"}}, status_code=500)

    spy.set_responder(responder)

    body = {"model": "m", "messages": [{"role": "user", "content": "x"}]}
    resp = proxy_client.post(CHAT_COMPLETIONS_PATH, json=body)
    assert resp.status_code == 500
    assert resp.json()["error"]["message"] == "boom"

    resp_files = list(trace_dir.glob("*.response.json"))
    assert len(resp_files) == 1
    record = json.loads(resp_files[0].read_text())
    assert record["status_code"] == 500
    assert record["body"]["error"]["message"] == "boom"


def test_request_id_is_consistent_across_request_and_response_files(
    spy: UpstreamSpy, trace_dir: Path, proxy_client: TestClient
):
    async def responder(request):
        return JSONResponse({"choices": []})

    spy.set_responder(responder)

    proxy_client.post(
        CHAT_COMPLETIONS_PATH,
        json={"model": "m", "messages": [{"role": "user", "content": "."}]},
    )
    req_files = list(trace_dir.glob("*.request.json"))
    resp_files = list(trace_dir.glob("*.response.json"))
    assert len(req_files) == 1
    assert len(resp_files) == 1
    rid_from_req = json.loads(req_files[0].read_text())["request_id"]
    rid_from_resp = json.loads(resp_files[0].read_text())["request_id"]
    assert rid_from_req == rid_from_resp
    # Filenames also share the prefix
    assert req_files[0].name.startswith(rid_from_req)
    assert resp_files[0].name.startswith(rid_from_req)


def test_trace_dir_is_created_if_missing(tmp_path: Path, spy: UpstreamSpy):
    """make_app should create the trace dir on init."""
    target = tmp_path / "does" / "not" / "exist"
    upstream_app = _build_upstream(spy)
    upstream_transport = httpx.ASGITransport(app=upstream_app)
    upstream_client = httpx.AsyncClient(
        transport=upstream_transport, base_url="http://upstream"
    )
    make_app("http://upstream", target, client=upstream_client)
    assert target.is_dir()
