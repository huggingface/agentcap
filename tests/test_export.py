"""Unit tests for ``agentcap.export``.

Captures + traces are now pushed to a paired ``-captures`` /
``-<agent>-traces`` dataset pair under a single HF Collection. The
tests assert: URI parsing, repo-id derivation, the parquet payload
shape (incl. the new ``run_id`` column), the raw-JSONL trace upload,
the README cross-links, and ``ensure_collection`` idempotency.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentcap.export import (
    _row,
    captures_repo_id,
    detect_model,
    detect_provider_columns,
    ensure_collection,
    export_local,
    parse_collection_base,
    push_agent_traces_dataset,
    push_captures_dataset,
    traces_repo_id_for,
)


def _write_capture(
    capture_dir: Path,
    rid: str,
    body: dict,
    response: dict,
    *,
    upstream_url: str = "http://127.0.0.1:8000",
    upstream_fingerprint: dict | None = None,
) -> None:
    (capture_dir / f"{rid}.request.json").write_text(
        json.dumps({
            "request_id": rid,
            "captured_at": 1000,
            "upstream_url": upstream_url,
            "body": body,
        })
    )
    (capture_dir / f"{rid}.response.json").write_text(
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
    capture = tmp_path / "capture"
    capture.mkdir()
    body = {"model": "google/gemma-4-E4B-it", "messages": []}
    _write_capture(capture, "a", body, {"choices": []})
    _write_capture(capture, "b", body, {"choices": []})
    assert detect_model(capture) == "google/gemma-4-E4B-it"


def test_detect_model_strips_revision_suffix(tmp_path: Path):
    capture = tmp_path / "capture"
    capture.mkdir()
    _write_capture(capture, "a", {"model": "google/gemma-4-E4B-it", "messages": []}, {})
    _write_capture(capture, "b", {"model": "google/gemma-4-E4B-it@main", "messages": []}, {})
    assert detect_model(capture) == "google/gemma-4-E4B-it"


def test_detect_model_raises_on_mixed_models(tmp_path: Path):
    capture = tmp_path / "capture"
    capture.mkdir()
    _write_capture(capture, "a", {"model": "google/gemma-4-E4B-it", "messages": []}, {})
    _write_capture(capture, "b", {"model": "Qwen/Qwen3-7B", "messages": []}, {})
    with pytest.raises(ValueError) as exc_info:
        detect_model(capture)
    assert "multiple models" in str(exc_info.value)


def test_detect_model_returns_none_on_empty_capture_dir(tmp_path: Path):
    capture = tmp_path / "capture"
    capture.mkdir()
    assert detect_model(capture) is None


def test_detect_model_returns_none_when_no_request_has_model_field(tmp_path: Path):
    capture = tmp_path / "capture"
    capture.mkdir()
    (capture / "rid.request.json").write_text(
        json.dumps({"request_id": "rid", "captured_at": 1, "body": {"messages": []}})
    )
    assert detect_model(capture) is None


# ---------------------------------------------------------------------------
# Provider derivation from the per-request upstream_url stamp
# ---------------------------------------------------------------------------


def test_detect_provider_columns_hostname_classification(tmp_path: Path):
    capture = tmp_path / "capture"
    capture.mkdir()
    _write_capture(
        capture, "rid", _BODY, {},
        upstream_url="http://127.0.0.1:8000",
    )
    cols = detect_provider_columns(capture)
    assert cols == {"provider": "local", "upstream_url": "http://127.0.0.1:8000"}


def test_detect_provider_columns_hf_router_sub_provider_refinement(tmp_path: Path):
    capture = tmp_path / "capture"
    capture.mkdir()
    _write_capture(
        capture, "rid",
        {"model": "meta-llama/Llama-3.3-70B:fireworks-ai", "messages": []},
        {},
        upstream_url="https://router.huggingface.co",
    )
    cols = detect_provider_columns(capture)
    assert cols["provider"] == "hf-router/fireworks-ai"
    assert cols["upstream_url"] == "https://router.huggingface.co"


def test_detect_provider_columns_empty_when_no_upstream_stamp(tmp_path: Path):
    capture = tmp_path / "capture"
    capture.mkdir()
    (capture / "rid.request.json").write_text(
        json.dumps({"request_id": "rid", "captured_at": 1, "body": _BODY})
    )
    assert detect_provider_columns(capture) == {}


# ---------------------------------------------------------------------------
# Collection-base parsing + repo-id derivation
# ---------------------------------------------------------------------------


def test_parse_collection_base_owner_and_base():
    owner, base = parse_collection_base("owner/my-collection")
    assert owner == "owner"
    assert base == "my-collection"


def test_parse_collection_base_strips_hf_datasets_prefix():
    owner, base = parse_collection_base("hf://datasets/owner/base")
    assert (owner, base) == ("owner", "base")


def test_parse_collection_base_rejects_subdir():
    """A third segment is ambiguous — the collection-base form is a
    single ``<base>``, not a ``<base>/<subdir>``."""
    with pytest.raises(ValueError, match="<owner>/<base>"):
        parse_collection_base("owner/base/extra")


def test_parse_collection_base_rejects_missing_name():
    with pytest.raises(ValueError, match="<owner>/<base>"):
        parse_collection_base("owner")


def test_repo_id_derivation():
    assert captures_repo_id("me", "sweep") == "me/sweep-captures"
    assert traces_repo_id_for("me", "sweep", "pi") == "me/sweep-pi-traces"
    assert traces_repo_id_for("me", "sweep", "hermes") == "me/sweep-hermes-traces"


# ---------------------------------------------------------------------------
# push_captures_dataset
# ---------------------------------------------------------------------------


def test_push_captures_creates_captures_repo(tmp_path: Path, fake_hf_api):
    capture = tmp_path / "capture"
    capture.mkdir()
    _write_capture(capture, "rid", _BODY, {"choices": []})

    repo_id, n_rows = push_captures_dataset(
        [{"capture_dir": capture, "model": "google/gemma-4-E4B-it", "agent": "pi",
          "run_id": "pi-local-20260601-090000"}],
        owner="me", base="sweep",
    )

    assert repo_id == "me/sweep-captures"
    assert n_rows == [1]
    assert fake_hf_api.created_repos[0] == {
        "repo_id": "me/sweep-captures", "repo_type": "dataset",
        "exist_ok": True, "private": True,
    }


def test_push_captures_lands_under_data(tmp_path: Path, fake_hf_api):
    import re

    capture = tmp_path / "capture"
    capture.mkdir()
    _write_capture(capture, "rid", _BODY, {"choices": []})

    push_captures_dataset(
        [{"capture_dir": capture, "model": "google/gemma-4-E4B-it",
          "agent": "pi", "run_id": "pi-local-20260601-090000"}],
        owner="me", base="sweep",
    )
    op = fake_hf_api.commits[0]["operations"][0]
    # ``-captures`` repo, single ``data/<filename>.parquet`` layout.
    assert re.fullmatch(
        r"data/train-pi-gemma-4-E4B-it-local-\d{8}T\d{6}-[0-9a-f]{6}\.parquet",
        op["path_in_repo"],
    ), op["path_in_repo"]


def test_push_captures_stamps_run_id_column(tmp_path: Path, fake_hf_api):
    capture = tmp_path / "capture"
    capture.mkdir()
    _write_capture(capture, "rid", _BODY, {"choices": []})

    push_captures_dataset(
        [{"capture_dir": capture, "model": "m", "agent": "pi",
          "run_id": "pi-local-20260601-090000"}],
        owner="me", base="sweep",
    )
    op = fake_hf_api.commits[0]["operations"][0]
    assert "run_id" in op["columns"]


def test_push_captures_batches_into_one_commit(tmp_path: Path, fake_hf_api):
    items = []
    for i in range(3):
        cap = tmp_path / f"capture-{i}"
        cap.mkdir()
        _write_capture(cap, f"rid{i}", _BODY, {})
        items.append({
            "capture_dir": cap, "model": "m", "agent": "hermes",
            "run_id": f"hermes-local-2026060{i+1}-000000",
        })

    push_captures_dataset(items, owner="me", base="sweep")
    assert len(fake_hf_api.commits) == 1
    assert len(fake_hf_api.commits[0]["operations"]) == 3
    paths = [op["path_in_repo"] for op in fake_hf_api.commits[0]["operations"]]
    assert len(set(paths)) == 3, f"filenames collided: {paths}"


def test_push_captures_seeds_readme_with_collection_link(
    tmp_path: Path, fake_hf_api,
):
    fake_hf_api.existing_files = []  # simulate freshly-created repo
    capture = tmp_path / "capture"
    capture.mkdir()
    _write_capture(capture, "rid", _BODY, {})

    push_captures_dataset(
        [{"capture_dir": capture, "model": "m", "run_id": "r"}],
        owner="me", base="sweep",
    )

    ops = fake_hf_api.commits[0]["operations"]
    readme_ops = [op for op in ops if op["path_in_repo"] == "README.md"]
    assert readme_ops, "captures README missing on first push"
    body = readme_ops[0]["bytes"].decode("utf-8")
    # Cross-links to the traces sibling family and the Collection.
    assert "me/sweep-captures" in body
    assert "sweep-<agent>-traces" in body
    assert "sweep Collection" in body


def test_push_captures_skips_readme_on_subsequent_push(
    tmp_path: Path, fake_hf_api,
):
    # fake_hf_api defaults to existing_files=["README.md"]
    capture = tmp_path / "capture"
    capture.mkdir()
    _write_capture(capture, "rid", _BODY, {})

    push_captures_dataset(
        [{"capture_dir": capture, "model": "m", "run_id": "r"}],
        owner="me", base="sweep",
    )
    paths = [op["path_in_repo"] for op in fake_hf_api.commits[0]["operations"]]
    assert "README.md" not in paths


# ---------------------------------------------------------------------------
# push_agent_traces_dataset — raw JSONL upload
# ---------------------------------------------------------------------------


def test_push_traces_uploads_files_as_is(tmp_path: Path, fake_hf_api):
    fake_hf_api.existing_files = []
    traces = tmp_path / "traces"
    traces.mkdir()
    (traces / "session-a.jsonl").write_text('{"type":"session","id":"a"}\n')
    (traces / "session-b.jsonl").write_text('{"type":"session","id":"b"}\n')

    repo_id, n_files = push_agent_traces_dataset(
        [{"traces_dir": traces, "run_id": "pi-local-20260601-090000"}],
        owner="me", base="sweep", agent="pi",
    )

    assert repo_id == "me/sweep-pi-traces"
    assert n_files == 2
    paths = [op["path_in_repo"] for op in fake_hf_api.commits[0]["operations"]]
    # One README + two raw files under data/<run_id>/.
    assert "README.md" in paths
    assert "data/pi-local-20260601-090000/session-a.jsonl" in paths
    assert "data/pi-local-20260601-090000/session-b.jsonl" in paths


def test_push_traces_readme_marks_agent_and_links_captures(
    tmp_path: Path, fake_hf_api,
):
    fake_hf_api.existing_files = []
    traces = tmp_path / "traces"
    traces.mkdir()
    (traces / "x.jsonl").write_text("{}\n")

    push_agent_traces_dataset(
        [{"traces_dir": traces, "run_id": "r1"}],
        owner="me", base="sweep", agent="hermes",
    )
    ops = fake_hf_api.commits[0]["operations"]
    readme = next(op for op in ops if op["path_in_repo"] == "README.md")
    body = readme["bytes"].decode("utf-8")
    # Tags: agent-traces, agentcap-traces, per-agent suffix.
    assert "agent-traces" in body
    assert "agentcap-traces-hermes" in body
    # source_datasets points back at the captures sibling.
    assert "me/sweep-captures" in body
    assert "sweep Collection" in body


def test_push_traces_skips_when_no_files_and_readme_exists(
    tmp_path: Path, fake_hf_api,
):
    """Empty trace dir + README already in repo → no commit."""
    traces = tmp_path / "traces"
    traces.mkdir()
    repo_id, n_files = push_agent_traces_dataset(
        [{"traces_dir": traces, "run_id": "r1"}],
        owner="me", base="sweep", agent="pi",
    )
    assert repo_id == "me/sweep-pi-traces"
    assert n_files == 0
    assert fake_hf_api.commits == []


def test_push_traces_repo_created_private(tmp_path: Path, fake_hf_api):
    traces = tmp_path / "traces"
    traces.mkdir()
    (traces / "x.jsonl").write_text("{}")
    push_agent_traces_dataset(
        [{"traces_dir": traces, "run_id": "r1"}],
        owner="me", base="sweep", agent="pi",
    )
    record = next(
        r for r in fake_hf_api.created_repos if r["repo_id"] == "me/sweep-pi-traces"
    )
    assert record["private"] is True


# ---------------------------------------------------------------------------
# ensure_collection — find-or-create + idempotent item-add
# ---------------------------------------------------------------------------


def test_ensure_collection_creates_when_missing(fake_hf_api):
    slug = ensure_collection(
        owner="me", base="sweep",
        repos=["me/sweep-captures", "me/sweep-pi-traces"],
    )
    assert slug.startswith("me/sweep-")
    assert len(fake_hf_api.collections_created) == 1
    assert fake_hf_api.collections_created[0]["title"] == "sweep"
    assert fake_hf_api.collections_created[0]["private"] is True
    item_ids = [it["item_id"] for it in fake_hf_api.collection_items]
    assert item_ids == ["me/sweep-captures", "me/sweep-pi-traces"]


def test_ensure_collection_is_idempotent_on_second_call(fake_hf_api):
    first = ensure_collection(
        owner="me", base="sweep",
        repos=["me/sweep-captures"],
    )
    second = ensure_collection(
        owner="me", base="sweep",
        repos=["me/sweep-captures", "me/sweep-hermes-traces"],
    )
    assert first == second
    # Only one collection was created across the two calls.
    assert len(fake_hf_api.collections_created) == 1


# ---------------------------------------------------------------------------
# Round-trip — captures parquet shape (incl. run_id column)
# ---------------------------------------------------------------------------


def test_export_local_round_trip(tmp_path: Path):
    """End-to-end: write captures, export with provider+run_id columns,
    read parquet back, assert columns + that request JSON survives
    serialisation."""
    import pyarrow.parquet as pq

    capture = tmp_path / "capture"
    capture.mkdir()
    _write_capture(
        capture, "ra", _BODY, {"choices": [{"index": 0}]},
        upstream_fingerprint={"x_served_by": "pod-7", "served_model": "gemma"},
    )
    _write_capture(capture, "rb", _BODY, {"choices": [{"index": 0}]})

    out = tmp_path / "rows.parquet"
    extra_cols = {
        "provider": "local",
        "upstream_url": "http://127.0.0.1:8000",
        "run_id": "pi-local-20260601-090000",
    }
    n_rows = export_local(
        capture, out, progress=False, provider_columns=extra_cols,
    )
    assert n_rows == 2

    table = pq.read_table(out)
    assert table.num_rows == 2
    assert set(table.column_names) == {
        "request_id", "model", "captured_at", "request", "response",
        "served_by", "served_build_info", "served_model",
        "provider", "upstream_url", "run_id",
    }
    rows = table.to_pylist()
    by_rid = {r["request_id"]: r for r in rows}
    assert by_rid["ra"]["served_by"] == "pod-7"
    assert by_rid["ra"]["served_model"] == "gemma"
    assert by_rid["rb"]["served_by"] is None
    for r in rows:
        assert r["provider"] == "local"
        assert r["upstream_url"] == "http://127.0.0.1:8000"
        assert r["run_id"] == "pi-local-20260601-090000"
    sample = json.loads(by_rid["ra"]["request"])
    assert sample["messages"][0]["role"] == "user"
