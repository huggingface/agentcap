"""Unit tests for ``agentcap.export``.

The export layer is a pure data shuffle — no rendering, no tokenization
— so the tests only need to assert that captured request/response files
in the capture dir come out as parquet rows with the expected metadata
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
    parse_dataset_uri,
    push_dataset,
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
# Dataset URI parsing + push_dataset
# ---------------------------------------------------------------------------


def test_parse_dataset_uri_with_subdir():
    repo_id, subdir = parse_dataset_uri("owner/name/runs/x/y")
    assert repo_id == "owner/name"
    assert subdir == "runs/x/y"


def test_parse_dataset_uri_no_subdir():
    repo_id, subdir = parse_dataset_uri("owner/name")
    assert repo_id == "owner/name"
    assert subdir == ""


def test_parse_dataset_uri_accepts_hf_datasets_prefix():
    repo_id, subdir = parse_dataset_uri("hf://datasets/owner/name/runs")
    assert repo_id == "owner/name"
    assert subdir == "runs"


def test_parse_dataset_uri_rejects_missing_name():
    with pytest.raises(ValueError, match="<owner>/<name>"):
        parse_dataset_uri("owner")


def test_push_dataset_uploads_under_data_subdir(tmp_path: Path, fake_hf_api):
    import re

    capture = tmp_path / "capture"
    capture.mkdir()
    _write_capture(capture, "rid1", _BODY, {"choices": []})
    _write_capture(capture, "rid2", _BODY, {"choices": []})

    fake = fake_hf_api

    push_dataset(
        [{"capture_dir": capture, "model": "google/gemma-4-E4B-it"}],
        "me/my-dataset/runs/abc",
    )

    assert fake.created_repo == {
        "repo_id": "me/my-dataset", "repo_type": "dataset", "exist_ok": True,
    }
    assert len(fake.commits) == 1
    commit = fake.commits[0]
    assert commit["repo_id"] == "me/my-dataset"
    assert commit["repo_type"] == "dataset"
    assert len(commit["operations"]) == 1
    op = commit["operations"][0]
    assert re.fullmatch(
        r"data/runs/abc/train-gemma-4-E4B-it-local-\d{8}T\d{6}-[0-9a-f]{6}\.parquet",
        op["path_in_repo"],
    ), op["path_in_repo"]
    assert op["n_rows"] == 2
    assert sorted(op["request_ids"]) == ["rid1", "rid2"]
    assert set(op["columns"]) == {
        "request_id", "model", "captured_at", "request", "response",
        "served_by", "served_build_info", "served_model",
        "provider", "upstream_url",
    }


def test_push_dataset_batches_multiple_items_into_one_commit(
    tmp_path: Path, fake_hf_api
):
    """N items → 1 create_commit call with N operations."""
    items = []
    for i in range(3):
        cap = tmp_path / f"capture-{i}"
        cap.mkdir()
        _write_capture(cap, f"rid{i}", _BODY, {})
        items.append({"capture_dir": cap, "model": "m", "agent": "hermes"})

    fake = fake_hf_api
    n_rows = push_dataset(items, "me/d/runs")

    assert n_rows == [1, 1, 1]
    assert len(fake.commits) == 1
    assert len(fake.commits[0]["operations"]) == 3
    paths = [op["path_in_repo"] for op in fake.commits[0]["operations"]]
    assert len(set(paths)) == 3, f"filenames collided: {paths}"


def test_push_dataset_no_subdir_lands_directly_under_data(
    tmp_path: Path, fake_hf_api
):
    capture = tmp_path / "capture"
    capture.mkdir()
    _write_capture(capture, "rid", _BODY, {})

    fake = fake_hf_api
    push_dataset([{"capture_dir": capture, "model": "m"}], "me/my-dataset")

    op = fake.commits[0]["operations"][0]
    assert op["path_in_repo"].startswith("data/train-")


def test_push_dataset_explicit_filename_overrides_default(
    tmp_path: Path, fake_hf_api
):
    capture = tmp_path / "capture"
    capture.mkdir()
    _write_capture(capture, "rid", _BODY, {})

    fake = fake_hf_api
    push_dataset(
        [{"capture_dir": capture, "model": "m", "filename": "latest.parquet"}],
        "me/my-dataset/runs",
    )
    assert fake.commits[0]["operations"][0]["path_in_repo"] == "data/runs/latest.parquet"


def test_push_dataset_default_filename_embeds_agent_and_slugs_model(
    tmp_path: Path, fake_hf_api
):
    import re

    capture = tmp_path / "capture"
    capture.mkdir()
    _write_capture(capture, "rid", _BODY, {})

    fake = fake_hf_api
    push_dataset(
        [{
            "capture_dir": capture,
            "model": "google/gemma-4-E4B-it",
            "agent": "goose",
        }],
        "me/my-dataset/runs",
    )
    op = fake.commits[0]["operations"][0]
    assert re.fullmatch(
        r"data/runs/train-goose-gemma-4-E4B-it-local-\d{8}T\d{6}-[0-9a-f]{6}\.parquet",
        op["path_in_repo"],
    ), op["path_in_repo"]


def test_push_dataset_seeds_readme_on_first_push(
    tmp_path: Path, fake_hf_api
):
    """First push (empty repo) bundles a README.md in the same commit
    as the parquet."""
    fake_hf_api.existing_files = []  # simulate freshly-created repo
    capture = tmp_path / "capture"
    capture.mkdir()
    _write_capture(capture, "rid", _BODY, {})

    push_dataset([{"capture_dir": capture, "model": "m"}], "me/my-dataset")

    ops = fake_hf_api.commits[0]["operations"]
    paths = [op["path_in_repo"] for op in ops]
    assert "README.md" in paths
    readme_op = next(op for op in ops if op["path_in_repo"] == "README.md")
    body = readme_op["bytes"].decode("utf-8")
    assert "me/my-dataset" in body
    assert "load_dataset" in body


def test_push_dataset_skips_readme_when_one_exists(
    tmp_path: Path, fake_hf_api
):
    """Subsequent pushes (README already in the repo) don't include it
    in the commit — user edits are preserved."""
    # fake_hf_api defaults to existing_files=["README.md"]
    capture = tmp_path / "capture"
    capture.mkdir()
    _write_capture(capture, "rid", _BODY, {})

    push_dataset([{"capture_dir": capture, "model": "m"}], "me/my-dataset")

    paths = [op["path_in_repo"] for op in fake_hf_api.commits[0]["operations"]]
    assert "README.md" not in paths


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def test_export_local_round_trip(tmp_path: Path):
    """End-to-end: write captures, export, read back, assert columns +
    that request JSON survives serialisation."""
    import pyarrow.parquet as pq

    capture = tmp_path / "capture"
    capture.mkdir()
    _write_capture(
        capture, "ra", _BODY, {"choices": [{"index": 0}]},
        upstream_fingerprint={"x_served_by": "pod-7", "served_model": "gemma"},
    )
    _write_capture(capture, "rb", _BODY, {"choices": [{"index": 0}]})

    out = tmp_path / "rows.parquet"
    n_rows = export_local(capture, out, progress=False)
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
