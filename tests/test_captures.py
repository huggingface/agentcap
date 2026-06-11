"""Unit tests for ``agentcap.captures``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentcap.captures import (
    load_request,
    load_requests,
    resolve_workspace_rid,
)


def _write_capture(d: Path, rid: str, body: dict) -> None:
    (d / f"{rid}.request.json").write_text(
        json.dumps({
            "request_id": rid,
            "captured_at": 1,
            "upstream_url": "http://localhost:8000",
            "body": body,
        })
    )


def test_load_request_from_capture_dir(tmp_path: Path) -> None:
    cap = tmp_path / "captures"
    cap.mkdir()
    body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    _write_capture(cap, "abc", body)

    assert load_request(str(cap), "abc") == body


def test_load_requests_batch_from_capture_dir(tmp_path: Path) -> None:
    cap = tmp_path / "captures"
    cap.mkdir()
    _write_capture(cap, "a", {"model": "m", "messages": []})
    _write_capture(cap, "b", {"model": "m", "messages": [{"role": "user"}]})

    out = load_requests(str(cap), ["a", "b"])
    assert set(out) == {"a", "b"}
    assert out["a"]["messages"] == []


def test_load_request_missing_id_raises(tmp_path: Path) -> None:
    cap = tmp_path / "captures"
    cap.mkdir()
    _write_capture(cap, "a", {"model": "m"})

    with pytest.raises(KeyError):
        load_request(str(cap), "ghost")


def test_load_request_from_parquet(tmp_path: Path) -> None:
    """Round-trip a body through ``export_local`` and back via the loader."""
    from agentcap.export import export_local

    cap = tmp_path / "captures"
    cap.mkdir()
    body = {
        "model": "m",
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [],
    }
    _write_capture(cap, "rid", body)
    # Pair with a minimal response file so export_local has both halves.
    (cap / "rid.response.json").write_text(json.dumps({
        "request_id": "rid", "captured_at_resp": 2,
        "status_code": 200, "body": {"choices": []},
    }))

    parquet = tmp_path / "out.parquet"
    n = export_local(cap, parquet, progress=False)
    assert n == 1

    loaded = load_request(str(parquet), "rid")
    assert loaded == body


def test_load_requests_bad_source(tmp_path: Path) -> None:
    not_a_thing = tmp_path / "nope.txt"
    not_a_thing.write_text("x")
    with pytest.raises(ValueError):
        load_requests(str(not_a_thing), ["a"])


def test_resolve_workspace_rid_finds_run(tmp_path: Path) -> None:
    ws = tmp_path / ".agentcap"
    run = ws / "hermes-local-20260101-000000"
    cap = run / "captures"
    cap.mkdir(parents=True)
    _write_capture(cap, "rid-target", {"model": "m"})

    found = resolve_workspace_rid(ws, "rid-target")
    assert found == (cap, "rid-target")


def test_resolve_workspace_rid_accepts_prefix(tmp_path: Path) -> None:
    ws = tmp_path / ".agentcap"
    cap = ws / "hermes-local-20260101-000000" / "captures"
    cap.mkdir(parents=True)
    _write_capture(cap, "abc12345deadbeef", {"model": "m"})

    found = resolve_workspace_rid(ws, "abc12345")
    assert found == (cap, "abc12345deadbeef")


def test_resolve_workspace_rid_ambiguous_prefix_raises(tmp_path: Path) -> None:
    from agentcap.captures import AmbiguousRequestId

    ws = tmp_path / ".agentcap"
    cap = ws / "hermes-local-20260101-000000" / "captures"
    cap.mkdir(parents=True)
    _write_capture(cap, "abc12345_a", {"model": "m"})
    _write_capture(cap, "abc12345_b", {"model": "m"})

    with pytest.raises(AmbiguousRequestId):
        resolve_workspace_rid(ws, "abc12345")


def test_resolve_workspace_rid_returns_none_when_absent(tmp_path: Path) -> None:
    ws = tmp_path / ".agentcap"
    ws.mkdir()
    assert resolve_workspace_rid(ws, "ghost") is None
