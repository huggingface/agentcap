"""Unit tests for ``agentcap.export``.

The export layer is a pure data shuffle — no rendering, no tokenization
— so the tests only need to assert that captured request/response files
in the trace dir come out as parquet rows with the expected metadata
columns. No fake processor needed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentcap.export import (
    _row,
    detect_model,
    detect_provider_columns,
    export_local,
    parse_bucket_uri,
    push_bucket,
)


def _write_capture(
    trace_dir: Path,
    rid: str,
    body: dict,
    response: dict,
    *,
    upstream_url: str = "http://127.0.0.1:8000",
    upstream_fingerprint: dict | None = None,
) -> None:
    (trace_dir / f"{rid}.request.json").write_text(
        json.dumps({
            "request_id": rid,
            "captured_at": 1000,
            "upstream_url": upstream_url,
            "body": body,
        })
    )
    (trace_dir / f"{rid}.response.json").write_text(
        json.dumps({
            "request_id": rid,
            "captured_at_resp": 1001,
            "stream": False,
            "status_code": 200,
            "body": response,
            "upstream_fingerprint": upstream_fingerprint or {},
        })
    )


_BODY = {
    "model": "google/gemma-4-E4B-it",
    "messages": [{"role": "user", "content": "u"}],
    "tools": [],
}


# ---------------------------------------------------------------------------
# Row construction
# ---------------------------------------------------------------------------


def test_row_serialises_bodies_as_json_strings():
    row = _row(
        request_id="rid",
        request_body={"model": "m", "messages": [{"role": "user", "content": "x"}]},
        response_body={"choices": [{"message": {"content": "hi"}}]},
        captured_at=42,
        upstream_fp=None,
    )
    assert row["request_id"] == "rid"
    assert row["model"] == "m"
    assert row["captured_at"] == 42
    # request/response are JSON strings, not dicts.
    assert isinstance(row["request"], str)
    assert isinstance(row["response"], str)
    assert json.loads(row["request"])["messages"][0]["content"] == "x"
    assert json.loads(row["response"])["choices"][0]["message"]["content"] == "hi"


def test_row_includes_fingerprint_columns_when_present():
    fp = {
        "x_served_by": "fireworks-pod-7",
        "build_info": "b9039",
        "served_model": "qwen-actually-served",
    }
    row = _row("rid", _BODY, {}, 1, fp)
    assert row["served_by"] == "fireworks-pod-7"
    assert row["served_build_info"] == "b9039"
    assert row["served_model"] == "qwen-actually-served"


def test_row_fingerprint_columns_default_to_none():
    row = _row("rid", _BODY, {}, 1, None)
    assert row["served_by"] is None
    assert row["served_build_info"] is None
    assert row["served_model"] is None


def test_row_empty_response_serialises_to_empty_object():
    row = _row("rid", _BODY, None, 1, None)
    assert row["response"] == "{}"


# ---------------------------------------------------------------------------
# detect_model — same uniqueness contract as before
# ---------------------------------------------------------------------------


def test_detect_model_returns_unique_model(tmp_path: Path):
    trace = tmp_path / "trace"
    trace.mkdir()
    body = {"model": "google/gemma-4-E4B-it", "messages": []}
    _write_capture(trace, "a", body, {"choices": []})
    _write_capture(trace, "b", body, {"choices": []})
    assert detect_model(trace) == "google/gemma-4-E4B-it"


def test_detect_model_strips_revision_suffix(tmp_path: Path):
    trace = tmp_path / "trace"
    trace.mkdir()
    _write_capture(trace, "a", {"model": "google/gemma-4-E4B-it", "messages": []}, {})
    _write_capture(trace, "b", {"model": "google/gemma-4-E4B-it@main", "messages": []}, {})
    assert detect_model(trace) == "google/gemma-4-E4B-it"


def test_detect_model_raises_on_mixed_models(tmp_path: Path):
    trace = tmp_path / "trace"
    trace.mkdir()
    _write_capture(trace, "a", {"model": "google/gemma-4-E4B-it", "messages": []}, {})
    _write_capture(trace, "b", {"model": "Qwen/Qwen3-7B", "messages": []}, {})
    with pytest.raises(ValueError) as exc_info:
        detect_model(trace)
    assert "multiple models" in str(exc_info.value)


def test_detect_model_returns_none_on_empty_trace_dir(tmp_path: Path):
    trace = tmp_path / "trace"
    trace.mkdir()
    assert detect_model(trace) is None


def test_detect_model_returns_none_when_no_request_has_model_field(tmp_path: Path):
    trace = tmp_path / "trace"
    trace.mkdir()
    (trace / "rid.request.json").write_text(
        json.dumps({"request_id": "rid", "captured_at": 1, "body": {"messages": []}})
    )
    assert detect_model(trace) is None


# ---------------------------------------------------------------------------
# Provider derivation from the per-request upstream_url stamp
# ---------------------------------------------------------------------------


def test_detect_provider_columns_hostname_classification(tmp_path: Path):
    trace = tmp_path / "trace"
    trace.mkdir()
    _write_capture(
        trace, "rid", _BODY, {},
        upstream_url="http://127.0.0.1:8000",
    )
    cols = detect_provider_columns(trace)
    assert cols == {"provider": "local", "upstream_url": "http://127.0.0.1:8000"}


def test_detect_provider_columns_hf_router_sub_provider_refinement(tmp_path: Path):
    trace = tmp_path / "trace"
    trace.mkdir()
    _write_capture(
        trace, "rid",
        {"model": "meta-llama/Llama-3.3-70B:fireworks-ai", "messages": []},
        {},
        upstream_url="https://router.huggingface.co",
    )
    cols = detect_provider_columns(trace)
    assert cols["provider"] == "hf-router/fireworks-ai"
    assert cols["upstream_url"] == "https://router.huggingface.co"


def test_detect_provider_columns_empty_when_no_upstream_stamp(tmp_path: Path):
    trace = tmp_path / "trace"
    trace.mkdir()
    (trace / "rid.request.json").write_text(
        json.dumps({"request_id": "rid", "captured_at": 1, "body": _BODY})
    )
    assert detect_provider_columns(trace) == {}


# ---------------------------------------------------------------------------
# Bucket URI parsing + push_bucket
# ---------------------------------------------------------------------------


def test_parse_bucket_uri_with_prefix():
    bucket_id, path = parse_bucket_uri("hf://buckets/owner/name/runs/x/y")
    assert bucket_id == "owner/name"
    assert path == "runs/x/y"


def test_parse_bucket_uri_no_prefix():
    bucket_id, path = parse_bucket_uri("hf://buckets/owner/name")
    assert bucket_id == "owner/name"
    assert path == ""


def test_parse_bucket_uri_rejects_non_bucket():
    with pytest.raises(ValueError, match="not a bucket URI"):
        parse_bucket_uri("hf://datasets/owner/name")


def test_parse_bucket_uri_rejects_missing_name():
    with pytest.raises(ValueError, match="hf://buckets/<owner>/<name>"):
        parse_bucket_uri("hf://buckets/owner")


def test_push_bucket_writes_parquet_to_prefix(tmp_path: Path, monkeypatch):
    import re

    import pyarrow.parquet as pq

    trace = tmp_path / "trace"
    trace.mkdir()
    _write_capture(trace, "rid1", _BODY, {"choices": []})
    _write_capture(trace, "rid2", _BODY, {"choices": []})

    captured: dict = {}

    def _fake_upload(bucket_id, *, add):
        captured["bucket_id"] = bucket_id
        captured["uploads"] = []
        for local, remote in add:
            table = pq.read_table(local)
            captured["uploads"].append({
                "remote": remote,
                "n_rows": table.num_rows,
                "columns": list(table.column_names),
                "request_ids": list(table.column("request_id").to_pylist()),
            })

    monkeypatch.setattr(
        "huggingface_hub.batch_bucket_files", _fake_upload, raising=False
    )

    push_bucket(
        trace, "hf://buckets/me/my-bucket/runs/abc",
        model="google/gemma-4-E4B-it",
    )

    assert captured["bucket_id"] == "me/my-bucket"
    assert len(captured["uploads"]) == 1
    upload = captured["uploads"][0]
    assert re.fullmatch(
        r"runs/abc/train-gemma-4-E4B-it-local-\d{8}T\d{6}-[0-9a-f]{6}\.parquet",
        upload["remote"],
    ), upload["remote"]
    assert upload["n_rows"] == 2
    assert sorted(upload["request_ids"]) == ["rid1", "rid2"]
    # The schema after the manifest drop — pure data shuffle plus
    # per-row fingerprint plus per-file provider columns.
    assert set(upload["columns"]) == {
        "request_id", "model", "captured_at", "request", "response",
        "served_by", "served_build_info", "served_model",
        "provider", "upstream_url",
    }


def test_push_bucket_default_filenames_are_unique_across_calls(
    tmp_path: Path, monkeypatch
):
    """Two consecutive pushes to the same prefix must not collide."""
    trace = tmp_path / "trace"
    trace.mkdir()
    _write_capture(trace, "rid", _BODY, {})

    seen_paths: list[str] = []

    def _fake_upload(bucket_id, *, add):
        for _, remote in add:
            seen_paths.append(remote)

    monkeypatch.setattr(
        "huggingface_hub.batch_bucket_files", _fake_upload, raising=False
    )

    for _ in range(3):
        push_bucket(trace, "hf://buckets/me/b/prefix", model="m")
    assert len(set(seen_paths)) == 3, f"filenames collided: {seen_paths}"


def test_push_bucket_explicit_filename_overrides_default(
    tmp_path: Path, monkeypatch
):
    trace = tmp_path / "trace"
    trace.mkdir()
    _write_capture(trace, "rid", _BODY, {})

    captured: dict = {}

    def _fake_upload(bucket_id, *, add):
        captured["remote_paths"] = [remote for _, remote in add]

    monkeypatch.setattr(
        "huggingface_hub.batch_bucket_files", _fake_upload, raising=False
    )

    push_bucket(
        trace, "hf://buckets/me/my-bucket/runs",
        model="m", filename="latest.parquet",
    )
    assert captured["remote_paths"] == ["runs/latest.parquet"]


def test_push_bucket_default_filename_embeds_agent_and_slugs_model(
    tmp_path: Path, monkeypatch
):
    import re

    trace = tmp_path / "trace"
    trace.mkdir()
    _write_capture(trace, "rid", _BODY, {})

    captured: dict = {}

    def _fake_upload(bucket_id, *, add):
        captured["remote_paths"] = [remote for _, remote in add]

    monkeypatch.setattr(
        "huggingface_hub.batch_bucket_files", _fake_upload, raising=False
    )

    push_bucket(
        trace, "hf://buckets/me/my-bucket/runs",
        model="google/gemma-4-E4B-it",
        agent="goose",
    )

    assert len(captured["remote_paths"]) == 1
    assert re.fullmatch(
        r"runs/train-goose-gemma-4-E4B-it-local-\d{8}T\d{6}-[0-9a-f]{6}\.parquet",
        captured["remote_paths"][0],
    ), captured["remote_paths"][0]


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def test_export_local_round_trip(tmp_path: Path):
    """End-to-end: write captures, export, read back, assert columns +
    that request JSON survives serialisation."""
    import pyarrow.parquet as pq

    trace = tmp_path / "trace"
    trace.mkdir()
    _write_capture(
        trace, "ra", _BODY, {"choices": [{"index": 0}]},
        upstream_fingerprint={"x_served_by": "pod-7", "served_model": "gemma"},
    )
    _write_capture(trace, "rb", _BODY, {"choices": [{"index": 0}]})

    out = tmp_path / "rows.parquet"
    n_rows = export_local(trace, out, progress=False)
    assert n_rows == 2

    table = pq.read_table(out)
    assert table.num_rows == 2
    assert set(table.column_names) == {
        "request_id", "model", "captured_at", "request", "response",
        "served_by", "served_build_info", "served_model",
        "provider", "upstream_url",
    }
    rows = table.to_pylist()
    by_rid = {r["request_id"]: r for r in rows}
    # Per-row fingerprint stamped from the captured response.
    assert by_rid["ra"]["served_by"] == "pod-7"
    assert by_rid["ra"]["served_model"] == "gemma"
    assert by_rid["rb"]["served_by"] is None
    # Per-file provider columns stamped from the upstream_url.
    for r in rows:
        assert r["provider"] == "local"
        assert r["upstream_url"] == "http://127.0.0.1:8000"
    # Round-trip the request JSON.
    sample = json.loads(by_rid["ra"]["request"])
    assert sample["messages"][0]["role"] == "user"
