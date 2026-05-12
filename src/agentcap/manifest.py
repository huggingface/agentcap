"""Offline manifest builder.

Given a captured chat-completion request body and a model tokenizer
(via the ``processor`` interface — anything with ``apply_chat_template``
that mimics transformers' ``AutoTokenizer`` / ``AutoProcessor``),
compute:

- per-message ``sections`` with ``tok_range``, role, stability, and
  (for ``role=tool``) ``tool_name`` + ``tool_call_id``.
- ``token_role``: per-token role label for the rendered prompt.
- ``rendered_tokens``: the integer token id sequence for the prompt,
  aligned with ``token_role``. Lets consumers do byte-stable matching /
  splice selection without loading a tokenizer or re-discovering the
  per-agent normalisation in :func:`_normalize_for_render` (Qwen3-Coder
  templates crash on raw OpenAI-spec input).

Hashable cache keys (prefix_id, args_hash, agent_build_id) stay
consumer-side — different consumers want different definitions.

The capture proxy never imports this module — manifest computation is
strictly export-side.
"""

from __future__ import annotations

import json
from typing import Any


def _flatten_content_parts(content):
    """Collapse OpenAI "content parts" lists into a plain string.

    Some clients send ``content`` as a list of typed parts
    (``[{"type": "text", "text": "..."}]``); some chat templates
    (Qwen3-Coder) try to string-concat the list and crash. Join the
    text parts in order; non-text parts are dropped. Non-list values
    pass through unchanged.
    """
    if not isinstance(content, list):
        return content
    out: list[str] = []
    for p in content:
        if isinstance(p, dict) and p.get("type") == "text":
            t = p.get("text")
            if isinstance(t, str):
                out.append(t)
    return "".join(out)


def _normalize_for_render(request_body: dict) -> dict:
    """Return a shallowly-rebuilt request body that real chat templates
    can render.

    Two normalisations, applied per message:

    - ``tool_calls[*].function.arguments`` parsed from JSON string to
      dict (OpenAI spec serialises it as a string; Qwen3-Coder's
      template iterates it as a mapping).
    - ``content`` lists of typed parts collapsed to a plain string
      (some clients use the multimodal-style list form; Qwen3-Coder's
      template tries to string-concat the list).

    The captured ``request`` column in the parquet stays byte-verbatim
    — this normalisation is render-only.
    """
    messages = request_body.get("messages") or []
    new_messages: list = []
    for m in messages:
        if not isinstance(m, dict):
            new_messages.append(m)
            continue
        new_m = {**m}
        if "content" in new_m:
            new_m["content"] = _flatten_content_parts(new_m["content"])
        calls = new_m.get("tool_calls")
        if calls:
            new_calls = []
            for tc in calls:
                fn = (tc.get("function") if isinstance(tc, dict) else None) or {}
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        parsed = json.loads(args)
                    except (json.JSONDecodeError, TypeError):
                        new_calls.append(tc)
                        continue
                    new_calls.append({**tc, "function": {**fn, "arguments": parsed}})
                else:
                    new_calls.append(tc)
            new_m["tool_calls"] = new_calls
        new_messages.append(new_m)
    return {**request_body, "messages": new_messages}


def _render_ids(
    processor,
    messages: list,
    tools: Any,
    *,
    add_generation_prompt: bool = False,
) -> list[int]:
    if not messages:
        return []
    out = processor.apply_chat_template(
        messages,
        tools=tools,
        tokenize=True,
        return_dict=True,
        add_generation_prompt=add_generation_prompt,
    )
    ids = out["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def _render_len(processor, messages: list, tools: Any) -> int:
    return len(_render_ids(processor, messages, tools))


def compute_sections(processor, request_body: dict) -> list[dict]:
    """One section per message in the captured request.

    Cumulative-length walk: render ``messages[:i+1]`` both with and
    without ``tools`` to derive ``tokens`` and ``tools_injection_tokens``
    per message.
    """
    messages = list(request_body.get("messages") or [])
    tools = request_body.get("tools")
    sections: list[dict] = []

    # Pre-scan for assistant-side tool_calls so we can attribute the
    # originating tool_name onto subsequent role=tool messages by
    # tool_call_id (the OpenAI-compat join key).
    tool_call_lookup: dict[str, str] = {}
    for m in messages:
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                tcid = tc.get("id")
                fn = tc.get("function") or {}
                if tcid:
                    tool_call_lookup[tcid] = fn.get("name", "")

    cum_with = 0
    cum_without = 0
    seen_non_system = False

    for i, m in enumerate(messages):
        new_with = _render_len(processor, messages[: i + 1], tools)
        new_without = _render_len(processor, messages[: i + 1], None)
        tokens = new_with - cum_with
        tokens_without_tools = new_without - cum_without
        tools_injection_tokens = tokens - tokens_without_tools

        role = m.get("role", "?")
        section: dict[str, Any] = {
            "id": f"msg-{i}-{role}",
            "role": role,
            "tok_range": [cum_with, new_with],
            "tokens": tokens,
            "tokens_without_tools": tokens_without_tools,
            "tools_injection_tokens": tools_injection_tokens,
            "stable": role == "system" and not seen_non_system,
        }

        if role == "tool":
            tcid = m.get("tool_call_id")
            section["tool_call_id"] = tcid
            if tcid and tcid in tool_call_lookup:
                section["tool_name"] = tool_call_lookup[tcid]
            else:
                section["tool_name"] = m.get("name") or ""

        sections.append(section)
        cum_with = new_with
        cum_without = new_without
        if role != "system":
            seen_non_system = True

    return sections


def per_token_roles(processor, request_body: dict) -> list[str]:
    """Per-token role label for the rendered prompt (with tools)."""
    messages = list(request_body.get("messages") or [])
    tools = request_body.get("tools")
    roles: list[str] = []
    prev = 0
    for i, m in enumerate(messages):
        cur = _render_len(processor, messages[: i + 1], tools)
        roles.extend([m.get("role", "?")] * (cur - prev))
        prev = cur
    return roles


def build_manifest(
    processor,
    *,
    model: str,
    request_id: str,
    captured_at: int,
    request_body: dict,
    response_body: dict | None = None,
) -> dict:
    """Assemble a single manifest row.

    Includes ``rendered_tokens`` (list[int], aligned with
    ``token_role``). Recomputing this consumer-side requires
    re-applying :func:`_normalize_for_render` (Qwen3-Coder templates
    crash on list-typed content / string-typed tool_call.arguments),
    which isn't a public API; shipping the ids removes that coupling
    at a ~10× row-size cost.
    """
    rendered_body = _normalize_for_render(request_body)
    messages = list(rendered_body.get("messages") or [])
    tools = rendered_body.get("tools")
    rendered_tokens = _render_ids(processor, messages, tools)
    sections = compute_sections(processor, rendered_body)
    token_role = per_token_roles(processor, rendered_body)
    n_tokens = sections[-1]["tok_range"][1] if sections else 0

    return {
        "request_id": request_id,
        "model": model,
        "captured_at": captured_at,
        "request": request_body,
        "response": response_body or {},
        "n_tokens": n_tokens,
        "sections": sections,
        "token_role": token_role,
        "rendered_tokens": rendered_tokens,
    }
