"""Unit tests for ``agentcap.manifest`` and ``agentcap.export``.

A ``FakeProcessor`` mimics ``transformers.AutoTokenizer.apply_chat_template``
deterministically — same input always yields same token-id list — so we
can assert exact tok_range / sections math without downloading a real
HF tokenizer.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentcap.manifest import (
    build_manifest,
    compute_sections,
    per_token_roles,
)
from agentcap.export import build_rows


# ---------------------------------------------------------------------------
# Fake processor
# ---------------------------------------------------------------------------


_TOOLS_HEADER_TOKENS = 7
_ROLE_HEADER_TOKENS = 2
_GEN_PROMPT_TOKENS = 1


class FakeProcessor:
    """Deterministic stand-in for AutoTokenizer.apply_chat_template.

    Token-count contract:
      - 1 token per character of ``content``.
      - +``_ROLE_HEADER_TOKENS`` per message (role markers).
      - +``_TOOLS_HEADER_TOKENS`` injected once at the top when ``tools``
        is non-empty.
      - +``_GEN_PROMPT_TOKENS`` if ``add_generation_prompt`` is True.
      - tool_calls on assistant messages: 1 token per char of
        ``json.dumps(arguments)``.

    Token IDs are not meaningful (all zeros) — only lengths matter.
    """

    def apply_chat_template(
        self,
        messages,
        tools=None,
        tokenize=True,
        return_dict=True,
        add_generation_prompt=False,
    ):
        ids: list[int] = []
        if tools:
            ids.extend([0] * _TOOLS_HEADER_TOKENS)
        for m in messages:
            ids.extend([0] * _ROLE_HEADER_TOKENS)
            content = m.get("content") or ""
            if isinstance(content, str):
                ids.extend([0] * len(content))
            else:
                ids.extend([0] * len(json.dumps(content)))
            for tc in m.get("tool_calls") or []:
                args = (tc.get("function") or {}).get("arguments", "")
                if not isinstance(args, str):
                    args = json.dumps(args)
                ids.extend([0] * len(args))
        if add_generation_prompt:
            ids.extend([0] * _GEN_PROMPT_TOKENS)
        return {"input_ids": [ids]}


# ---------------------------------------------------------------------------
# compute_sections
# ---------------------------------------------------------------------------


def test_sections_tok_ranges_are_contiguous_and_match_total():
    proc = FakeProcessor()
    body = {
        "messages": [
            {"role": "system", "content": "sysprompt"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ],
        "tools": [{"type": "function", "function": {"name": "ls"}}],
    }
    sections = compute_sections(proc, body)
    assert len(sections) == 3
    # Ranges abut with no gap and start at 0
    assert sections[0]["tok_range"][0] == 0
    for prev, nxt in zip(sections, sections[1:]):
        assert prev["tok_range"][1] == nxt["tok_range"][0]
    # Sum of section.tokens == total render length
    total = sum(s["tokens"] for s in sections)
    expected_total = (
        _TOOLS_HEADER_TOKENS
        + 3 * _ROLE_HEADER_TOKENS
        + len("sysprompt")
        + len("hello")
        + len("hi")
    )
    assert total == expected_total
    assert sections[-1]["tok_range"][1] == expected_total


def test_sections_tools_injection_is_credited_to_first_message():
    proc = FakeProcessor()
    body = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u"},
        ],
        "tools": [{"type": "function", "function": {"name": "ls"}}],
    }
    sections = compute_sections(proc, body)
    # First message takes the tools-header injection; second sees zero.
    assert sections[0]["tools_injection_tokens"] == _TOOLS_HEADER_TOKENS
    assert sections[1]["tools_injection_tokens"] == 0
    # tokens = tokens_without_tools + tools_injection_tokens, per section
    for s in sections:
        assert s["tokens"] == s["tokens_without_tools"] + s["tools_injection_tokens"]


def test_sections_stable_only_for_leading_system():
    proc = FakeProcessor()
    body = {
        "messages": [
            {"role": "system", "content": "a"},
            {"role": "system", "content": "b"},
            {"role": "user", "content": "u"},
            {"role": "system", "content": "c"},  # mid-conversation: not stable
            {"role": "assistant", "content": "x"},
        ],
        "tools": [],
    }
    sections = compute_sections(proc, body)
    assert sections[0]["stable"] is True
    assert sections[1]["stable"] is True
    assert sections[2]["stable"] is False
    assert sections[3]["stable"] is False  # role=system but after a user turn
    assert sections[4]["stable"] is False


def test_sections_tool_message_inherits_tool_name_from_originating_call():
    proc = FakeProcessor()
    body = {
        "messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "skill_view", "arguments": '{"path": "/x"}'},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "RESULT",
            },
        ],
        "tools": [],
    }
    sections = compute_sections(proc, body)
    tool_section = sections[-1]
    assert tool_section["role"] == "tool"
    assert tool_section["tool_call_id"] == "call_1"
    assert tool_section["tool_name"] == "skill_view"
    # Manifest does not bake derived hashes into sections; consumers can
    # hash the raw arguments from the request body if they want.
    assert "args_hash" not in tool_section


def test_sections_tool_message_unknown_call_id_falls_back_to_message_name():
    proc = FakeProcessor()
    body = {
        "messages": [
            {"role": "user", "content": "u"},
            {
                "role": "tool",
                "tool_call_id": "call_unmatched",
                "name": "fallback_name",
                "content": "x",
            },
        ],
        "tools": [],
    }
    sections = compute_sections(proc, body)
    last = sections[-1]
    assert last["tool_call_id"] == "call_unmatched"
    assert last["tool_name"] == "fallback_name"


def test_sections_empty_messages():
    proc = FakeProcessor()
    assert compute_sections(proc, {"messages": []}) == []


# ---------------------------------------------------------------------------
# per_token_roles
# ---------------------------------------------------------------------------


def test_per_token_roles_length_matches_render():
    proc = FakeProcessor()
    body = {
        "messages": [
            {"role": "system", "content": "sysprompt"},
            {"role": "user", "content": "hello"},
        ],
        "tools": [],
    }
    roles = per_token_roles(proc, body)
    expected_total = (
        2 * _ROLE_HEADER_TOKENS + len("sysprompt") + len("hello")
    )
    assert len(roles) == expected_total
    assert set(roles) == {"system", "user"}


# ---------------------------------------------------------------------------
# build_manifest
# ---------------------------------------------------------------------------


def test_build_manifest_top_level_fields():
    proc = FakeProcessor()
    body = {
        "messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
        ],
        "tools": [],
    }
    row = build_manifest(
        proc,
        model="my-model",
        request_id="rid-1",
        captured_at=123,
        request_body=body,
        response_body={"choices": []},
    )
    assert row["request_id"] == "rid-1"
    assert row["model"] == "my-model"
    assert row["captured_at"] == 123
    # Rows never carry derived metadata — no agent_build_id, no
    # rendered_tokens, no args_hash. Consumers compute their own from
    # the raw `request` body if they need them.
    assert "agent_build_id" not in row
    assert "rendered_tokens" not in row
    assert row["request"] == body
    assert row["response"] == {"choices": []}
    assert row["n_tokens"] == row["sections"][-1]["tok_range"][1]
    assert len(row["token_role"]) == row["n_tokens"]


# ---------------------------------------------------------------------------
# build_rows — trace dir → in-memory rows
# ---------------------------------------------------------------------------


def _write_capture(trace_dir: Path, rid: str, body: dict, response: dict) -> None:
    (trace_dir / f"{rid}.request.json").write_text(
        json.dumps({"request_id": rid, "captured_at": 1000, "body": body})
    )
    (trace_dir / f"{rid}.response.json").write_text(
        json.dumps(
            {
                "request_id": rid,
                "captured_at_resp": 1001,
                "stream": False,
                "status_code": 200,
                "body": response,
            }
        )
    )


def test_build_rows_one_row_per_capture(tmp_path: Path):
    trace = tmp_path / "trace"
    trace.mkdir()
    body = {
        "messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
        ],
        "tools": [],
    }
    _write_capture(trace, "aaa", body, {"choices": [{"index": 0}]})
    _write_capture(trace, "bbb", body, {"choices": [{"index": 0}]})

    rows = build_rows(trace, processor=FakeProcessor(), model="m")
    assert len(rows) == 2
    assert {r["request_id"] for r in rows} == {"aaa", "bbb"}
    for r in rows:
        assert r["model"] == "m"
        assert r["sections"][0]["role"] == "system"


def test_build_rows_handles_streaming_response(tmp_path: Path):
    """Streaming responses captured by the proxy carry the assembled
    SSE bytes under a ``stream: True`` envelope; ``build_rows`` must
    pass that through verbatim into the row's ``response`` field."""
    trace = tmp_path / "trace"
    trace.mkdir()
    rid = "stream-rid"
    (trace / f"{rid}.request.json").write_text(
        json.dumps(
            {
                "request_id": rid,
                "captured_at": 1,
                "body": {"messages": [{"role": "user", "content": "x"}], "tools": []},
            }
        )
    )
    (trace / f"{rid}.response.json").write_text(
        json.dumps(
            {
                "request_id": rid,
                "captured_at_resp": 2,
                "stream": True,
                "status_code": 200,
                "raw": "data: {...}\n\ndata: [DONE]\n\n",
            }
        )
    )

    rows = build_rows(trace, processor=FakeProcessor(), model="m")
    assert rows[0]["response"]["stream"] is True
    assert "[DONE]" in rows[0]["response"]["raw"]


def test_build_rows_request_without_response_file(tmp_path: Path):
    """Orphan request file (no response written) yields a row with an
    empty response dict — useful for incomplete captures."""
    trace = tmp_path / "trace"
    trace.mkdir()
    rid = "orphan"
    (trace / f"{rid}.request.json").write_text(
        json.dumps(
            {
                "request_id": rid,
                "captured_at": 1,
                "body": {"messages": [{"role": "user", "content": "x"}], "tools": []},
            }
        )
    )

    rows = build_rows(trace, processor=FakeProcessor(), model="m")
    assert len(rows) == 1
    assert rows[0]["response"] == {}


# ---------------------------------------------------------------------------
# detect_model
# ---------------------------------------------------------------------------


def test_detect_model_returns_unique_model(tmp_path: Path):
    from agentcap.export import detect_model

    trace = tmp_path / "trace"
    trace.mkdir()
    body = {"model": "google/gemma-4-E4B-it", "messages": [{"role": "user", "content": "x"}]}
    _write_capture(trace, "a", body, {"choices": []})
    _write_capture(trace, "b", body, {"choices": []})
    assert detect_model(trace) == "google/gemma-4-E4B-it"


def test_detect_model_strips_revision_suffix(tmp_path: Path):
    from agentcap.export import detect_model

    trace = tmp_path / "trace"
    trace.mkdir()
    body_a = {"model": "google/gemma-4-E4B-it", "messages": [{"role": "user", "content": "x"}]}
    body_b = {"model": "google/gemma-4-E4B-it@main", "messages": [{"role": "user", "content": "x"}]}
    _write_capture(trace, "a", body_a, {"choices": []})
    _write_capture(trace, "b", body_b, {"choices": []})
    # bare id vs @main should not be considered distinct
    assert detect_model(trace) == "google/gemma-4-E4B-it"


def test_detect_model_raises_on_mixed_models(tmp_path: Path):
    from agentcap.export import detect_model

    trace = tmp_path / "trace"
    trace.mkdir()
    _write_capture(
        trace, "a",
        {"model": "google/gemma-4-E4B-it", "messages": [{"role": "user", "content": "x"}]},
        {"choices": []},
    )
    _write_capture(
        trace, "b",
        {"model": "Qwen/Qwen3-7B", "messages": [{"role": "user", "content": "x"}]},
        {"choices": []},
    )
    with pytest.raises(ValueError) as exc_info:
        detect_model(trace)
    assert "multiple models" in str(exc_info.value)
    assert "google/gemma-4-E4B-it" in str(exc_info.value)
    assert "Qwen/Qwen3-7B" in str(exc_info.value)


def test_detect_model_returns_none_on_empty_trace_dir(tmp_path: Path):
    from agentcap.export import detect_model

    trace = tmp_path / "trace"
    trace.mkdir()
    # Empty / no-model traces aren't an error from detect_model itself —
    # the CLI raises only when --model is also missing. detect_model just
    # reports "couldn't infer" via None.
    assert detect_model(trace) is None


def test_detect_model_returns_none_when_no_request_has_model_field(tmp_path: Path):
    from agentcap.export import detect_model

    trace = tmp_path / "trace"
    trace.mkdir()
    (trace / "no-model.request.json").write_text(
        json.dumps({"request_id": "no-model", "captured_at": 1, "body": {"messages": []}})
    )
    assert detect_model(trace) is None


def test_detect_model_ignores_request_without_model_field(tmp_path: Path):
    from agentcap.export import detect_model

    trace = tmp_path / "trace"
    trace.mkdir()
    # First request has no model field; second does — should pick up the second.
    (trace / "no-model.request.json").write_text(
        json.dumps({"request_id": "no-model", "captured_at": 1, "body": {"messages": []}})
    )
    _write_capture(
        trace, "real",
        {"model": "google/gemma-4-E4B-it", "messages": [{"role": "user", "content": "x"}]},
        {"choices": []},
    )
    assert detect_model(trace) == "google/gemma-4-E4B-it"


_BODY = {"messages": [{"role": "user", "content": "u"}], "tools": []}


# ---------------------------------------------------------------------------
# Bucket URI parsing + push_bucket
# ---------------------------------------------------------------------------


def test_parse_bucket_uri_with_prefix():
    from agentcap.export import parse_bucket_uri
    bucket_id, path = parse_bucket_uri("hf://buckets/owner/name/runs/x/y")
    assert bucket_id == "owner/name"
    assert path == "runs/x/y"


def test_parse_bucket_uri_no_prefix():
    from agentcap.export import parse_bucket_uri
    bucket_id, path = parse_bucket_uri("hf://buckets/owner/name")
    assert bucket_id == "owner/name"
    assert path == ""


def test_parse_bucket_uri_rejects_non_bucket():
    from agentcap.export import parse_bucket_uri
    with pytest.raises(ValueError, match="not a bucket URI"):
        parse_bucket_uri("hf://datasets/owner/name")


def test_parse_bucket_uri_rejects_missing_name():
    from agentcap.export import parse_bucket_uri
    with pytest.raises(ValueError, match="hf://buckets/<owner>/<name>"):
        parse_bucket_uri("hf://buckets/owner")


def test_push_bucket_writes_parquet_to_prefix(tmp_path: Path, monkeypatch):
    """``push_bucket`` writes a real parquet file via Dataset.to_parquet
    and uploads it under ``<prefix>/<filename>`` via batch_bucket_files.
    The default filename is auto-generated per call."""
    import re

    import agentcap.export as export_mod
    import pyarrow.parquet as pq

    new_trace = tmp_path / "new"
    new_trace.mkdir()
    _write_capture(new_trace, "rid1", _BODY, {"choices": []})
    _write_capture(new_trace, "rid2", _BODY, {"choices": []})

    captured: dict = {}

    def _fake_upload(bucket_id, *, add):
        captured["bucket_id"] = bucket_id
        captured["uploads"] = []
        for local, remote in add:
            table = pq.read_table(local)
            captured["uploads"].append({
                "remote": remote,
                "n_rows": table.num_rows,
                "request_ids": list(table.column("request_id").to_pylist()),
            })

    monkeypatch.setattr(
        "huggingface_hub.batch_bucket_files", _fake_upload, raising=False
    )

    export_mod.push_bucket(
        new_trace, "hf://buckets/me/my-bucket/runs/abc",
        processor=FakeProcessor(), model="m",
    )

    assert captured["bucket_id"] == "me/my-bucket"
    assert len(captured["uploads"]) == 1
    upload = captured["uploads"][0]
    # Default filename: train-YYYYMMDDTHHMMSS-HEX6.parquet under the prefix
    assert re.fullmatch(
        r"runs/abc/train-\d{8}T\d{6}-[0-9a-f]{6}\.parquet",
        upload["remote"],
    ), upload["remote"]
    assert upload["n_rows"] == 2
    assert sorted(upload["request_ids"]) == ["rid1", "rid2"]


def test_push_bucket_default_filenames_are_unique_across_calls(
    tmp_path: Path, monkeypatch
):
    """Two consecutive pushes to the same prefix must NOT collide on
    filename — that was the design fix that prompted this default."""
    import agentcap.export as export_mod

    new_trace = tmp_path / "new"
    new_trace.mkdir()
    _write_capture(new_trace, "rid", _BODY, {"choices": []})

    seen_paths: list[str] = []

    def _fake_upload(bucket_id, *, add):
        for _, remote in add:
            seen_paths.append(remote)

    monkeypatch.setattr(
        "huggingface_hub.batch_bucket_files", _fake_upload, raising=False
    )

    for _ in range(3):
        export_mod.push_bucket(
            new_trace, "hf://buckets/me/b/prefix",
            processor=FakeProcessor(), model="m",
        )
    assert len(seen_paths) == 3
    assert len(set(seen_paths)) == 3, f"filenames collided: {seen_paths}"


def test_push_bucket_no_prefix_writes_to_root(tmp_path: Path, monkeypatch):
    """When the URI is just ``hf://buckets/owner/name`` (no path
    component), the parquet file lands at the bucket root."""
    import re

    import agentcap.export as export_mod

    new_trace = tmp_path / "new"
    new_trace.mkdir()
    _write_capture(new_trace, "rid", _BODY, {"choices": []})

    captured: dict = {}

    def _fake_upload(bucket_id, *, add):
        captured["bucket_id"] = bucket_id
        captured["remote_paths"] = [remote for _, remote in add]

    monkeypatch.setattr(
        "huggingface_hub.batch_bucket_files", _fake_upload, raising=False
    )

    export_mod.push_bucket(
        new_trace, "hf://buckets/me/my-bucket",
        processor=FakeProcessor(), model="m",
    )
    assert len(captured["remote_paths"]) == 1
    # No prefix → file at bucket root, default name pattern
    assert re.fullmatch(
        r"train-\d{8}T\d{6}-[0-9a-f]{6}\.parquet",
        captured["remote_paths"][0],
    )


def test_push_bucket_explicit_filename_overrides_default(
    tmp_path: Path, monkeypatch
):
    """An explicit ``filename`` opts back into deterministic
    overwrite-in-place behaviour (e.g. a 'latest' pointer file)."""
    import agentcap.export as export_mod

    new_trace = tmp_path / "new"
    new_trace.mkdir()
    _write_capture(new_trace, "rid", _BODY, {"choices": []})

    captured: dict = {}

    def _fake_upload(bucket_id, *, add):
        captured["remote_paths"] = [remote for _, remote in add]

    monkeypatch.setattr(
        "huggingface_hub.batch_bucket_files", _fake_upload, raising=False
    )

    export_mod.push_bucket(
        new_trace, "hf://buckets/me/my-bucket/runs",
        processor=FakeProcessor(), model="m",
        filename="latest.parquet",
    )
    assert captured["remote_paths"] == ["runs/latest.parquet"]


@pytest.mark.integration
def test_export_local_round_trip_real_parquet(tmp_path: Path):
    """End-to-end against the real `datasets` + parquet pipeline:
    ``export_local`` writes a parquet file that ``load_dataset`` reads
    back with the expected columns."""
    pytest.importorskip("datasets")
    from agentcap.export import export_local
    from datasets import load_dataset

    trace = tmp_path / "trace"
    trace.mkdir()
    body = {
        "messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
        ],
        "tools": [],
    }
    _write_capture(trace, "ra", body, {"choices": [{"index": 0}]})
    _write_capture(trace, "rb", body, {"choices": [{"index": 0}]})

    out = tmp_path / "rows.parquet"
    ds = export_local(trace, out, processor=FakeProcessor(), model="m")
    assert len(ds) == 2
    assert out.is_file()

    reloaded = load_dataset("parquet", data_files=str(out), split="train")
    assert len(reloaded) == 2
    assert set(reloaded.column_names) >= {
        "request_id",
        "model",
        "n_tokens",
        "sections",
        "token_role",
    }
