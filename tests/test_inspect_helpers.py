"""Unit tests for the inspect picker's parsing helpers.

Covers ``_decode_sse_response`` (OpenAI-compatible SSE → synthesized
assistant message) and ``_parse_fzf_terms`` (fzf query → list of
substrings to highlight). These functions are pure and live behind the
interactive picker, so they're easy to drift on without notice.
"""

from __future__ import annotations

import json

from agentcap.__main__ import _decode_sse_response
from agentcap.__main__ import _parse_fzf_terms


def _sse(*objs) -> str:
    """Assemble an SSE blob: one ``data: <json>`` line per object,
    plus a trailing ``data: [DONE]`` marker like real servers send."""
    return (
        "\n".join(f"data: {json.dumps(o)}" for o in objs)
        + "\ndata: [DONE]\n"
    )


def test_decode_sse_empty_returns_empty_message():
    out = _decode_sse_response("")
    assert out == {"content": "", "tool_calls": [], "finish_reason": None}


def test_decode_sse_concatenates_content_chunks():
    raw = _sse(
        {"choices": [{"delta": {"content": "Hello"}}]},
        {"choices": [{"delta": {"content": ", "}}]},
        {"choices": [{"delta": {"content": "world!"}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    )
    out = _decode_sse_response(raw)
    assert out["content"] == "Hello, world!"
    assert out["tool_calls"] == []
    assert out["finish_reason"] == "stop"


def test_decode_sse_merges_tool_call_argument_fragments():
    # First chunk for a tool call carries id + function.name; later
    # chunks accumulate ``arguments`` fragments under the same index.
    raw = _sse(
        {"choices": [{"delta": {"tool_calls": [{
            "index": 0, "id": "call_1", "type": "function",
            "function": {"name": "read", "arguments": ""},
        }]}}]},
        {"choices": [{"delta": {"tool_calls": [{
            "index": 0, "function": {"arguments": '{"path"'},
        }]}}]},
        {"choices": [{"delta": {"tool_calls": [{
            "index": 0, "function": {"arguments": ': "a.py"}'},
        }]}}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
    )
    out = _decode_sse_response(raw)
    assert out["content"] == ""
    assert out["tool_calls"] == [{
        "id": "call_1", "type": "function",
        "function": {"name": "read", "arguments": '{"path": "a.py"}'},
    }]
    assert out["finish_reason"] == "tool_calls"


def test_decode_sse_keeps_multiple_tool_calls_in_index_order():
    # Two parallel tool calls — index 1's first chunk arrives before
    # index 0's last; the decoder must still emit them sorted by index.
    raw = _sse(
        {"choices": [{"delta": {"tool_calls": [{
            "index": 0, "id": "c0",
            "function": {"name": "first", "arguments": "{"},
        }]}}]},
        {"choices": [{"delta": {"tool_calls": [{
            "index": 1, "id": "c1",
            "function": {"name": "second", "arguments": "{}"},
        }]}}]},
        {"choices": [{"delta": {"tool_calls": [{
            "index": 0, "function": {"arguments": "}"},
        }]}}]},
    )
    out = _decode_sse_response(raw)
    names = [tc["function"]["name"] for tc in out["tool_calls"]]
    ids = [tc["id"] for tc in out["tool_calls"]]
    args = [tc["function"]["arguments"] for tc in out["tool_calls"]]
    assert names == ["first", "second"]
    assert ids == ["c0", "c1"]
    assert args == ["{}", "{}"]


def test_decode_sse_skips_malformed_json_lines():
    # A garbled chunk in the middle must not abort the whole stream.
    raw = (
        'data: {"choices":[{"delta":{"content":"ok"}}]}\n'
        "data: {not json\n"
        'data: {"choices":[{"delta":{"content":"!"}}]}\n'
        "data: [DONE]\n"
    )
    out = _decode_sse_response(raw)
    assert out["content"] == "ok!"


def test_decode_sse_ignores_non_data_and_blank_lines():
    # Real streams interleave keep-alive comments (``: ping``) and
    # blank separators between events.
    raw = (
        ": keepalive\n"
        "\n"
        'data: {"choices":[{"delta":{"content":"x"}}]}\n'
        "\n"
        "event: end\n"
        'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n'
        "data: [DONE]\n"
    )
    out = _decode_sse_response(raw)
    assert out["content"] == "x"
    assert out["finish_reason"] == "stop"


def test_parse_fzf_terms_empty_query_returns_empty_list():
    assert _parse_fzf_terms("") == []
    assert _parse_fzf_terms("   ") == []


def test_parse_fzf_terms_plain_words():
    assert _parse_fzf_terms("alpha beta") == ["alpha", "beta"]


def test_parse_fzf_terms_strips_exact_match_quote():
    # ``'word`` → exact-match in fzf; the leading quote is a fzf
    # operator, not part of the substring to highlight.
    assert _parse_fzf_terms("'hf-cli") == ["hf-cli"]


def test_parse_fzf_terms_strips_anchors():
    # ``^`` (prefix) and ``$`` (suffix) are fzf anchors — neither is
    # part of the substring being matched.
    assert _parse_fzf_terms("^foo") == ["foo"]
    assert _parse_fzf_terms("bar$") == ["bar"]
    assert _parse_fzf_terms("^baz$") == ["baz"]


def test_parse_fzf_terms_drops_negated_terms():
    # ``!word`` excludes matches in fzf — nothing to colour for it.
    assert _parse_fzf_terms("keep !drop also") == ["keep", "also"]


def test_parse_fzf_terms_drops_bare_or_separator():
    # A bare ``|`` between two terms is fzf's OR — not a substring.
    assert _parse_fzf_terms("a | b") == ["a", "b"]


def test_parse_fzf_terms_handles_mixed_operators():
    out = _parse_fzf_terms("'exact ^prefix suffix$ !nope plain")
    assert out == ["exact", "prefix", "suffix", "plain"]
