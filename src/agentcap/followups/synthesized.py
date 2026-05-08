"""Synthesized follow-up strategy.

Sends ``(original_task, agent's last response)`` to a small synthesizer
LLM and uses the response as the next user message.

By design the synthesizer call **bypasses the capture proxy** — it
talks to the model server (or a different endpoint) directly. The
agent trace must remain a clean record of agent↔model interaction;
the synthesizer is just a way to produce realistic next user inputs.
"""

from __future__ import annotations

import json
from typing import Callable

from . import FollowUp


PROMPT_TEMPLATE = """\
You are a developer interacting with a coding agent. Given the agent's
last response, produce ONE short follow-up question or instruction
(<=30 words) that pushes the conversation forward. Don't ask the
agent to summarise; ask it to do or show something.

Original task:
<<<{task}>>>

Agent's last response:
<<<{response}>>>

Follow-up:
"""


def _default_call_synth(
    *,
    upstream: str,
    model: str,
    prompt: str,
    timeout: float | None,
) -> str:
    """Default OpenAI-compat chat-completion call."""
    import httpx

    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        # Reason-by-default models (Gemma-4, Qwen3.5+) burn the budget
        # in reasoning_content before the answer; an 80-token cap was
        # silently producing empty content + finish_reason="length".
        "max_tokens": 2048,
        "temperature": 0.7,
    }
    url = upstream.rstrip("/") + "/v1/chat/completions"
    resp = httpx.post(url, json=body, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            f"synthesizer response missing choices[0].message.content: "
            f"{json.dumps(data)[:200]}"
        ) from exc


class SynthesizedFollowUp(FollowUp):
    name = "synthesized"

    def __init__(
        self,
        *,
        upstream: str,
        model: str,
        timeout: float | None = 60,
        call: Callable[..., str] | None = None,
        prompt_template: str = PROMPT_TEMPLATE,
        fallback: str = "continue",
    ) -> None:
        """``upstream`` should point at the model server **directly**,
        not at the capture proxy. ``call`` is overridable for tests."""
        self.upstream = upstream
        self.model = model
        self.timeout = timeout
        self._call = call or _default_call_synth
        self.prompt_template = prompt_template
        self.fallback = fallback

    def next(self, *, original_task: str, last_response: str, turn: int) -> str:
        prompt = self.prompt_template.format(
            task=original_task, response=last_response
        )
        try:
            text = self._call(
                upstream=self.upstream,
                model=self.model,
                prompt=prompt,
                timeout=self.timeout,
            )
        except Exception:
            return self.fallback
        text = text.strip()
        return text or self.fallback
