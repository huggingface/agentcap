"""OpenCode driver.

Drives ``opencode run --format json`` non-interactively. The provider
config (proxy URL, ``minimal`` agent definition) is baked into the
per-agent image at ``~/.config/opencode/opencode.json`` — see
[containers/agentcap-opencode.Containerfile](
../../../containers/agentcap-opencode.Containerfile). The driver
passes the model id at the CLI (``--model local/<id>``); session
continuity is via ``--session`` on resume.

OpenCode emits NDJSON events on stdout when invoked with
``--format json``. ``text`` events carry assistant chunks; the
session id appears in every event as ``sessionID``.

Always launch from a real project dir — opencode hangs ≥30 min if
the model directs it to recursively glob from filesystem root.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Sequence

from . import AgentDriver, AgentTurn
from ..sandbox import Sandbox


_DEFAULT_PROVIDER_NAME = "local"


def _iter_events(stdout: str):
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def parse_response_text(stdout: str) -> str:
    """Concatenate ``text`` events from an opencode NDJSON stream."""
    parts: list[str] = []
    for obj in _iter_events(stdout):
        if obj.get("type") == "text" and isinstance(obj.get("text"), str):
            parts.append(obj["text"])
    return "".join(parts).strip()


def parse_session_id(stdout: str) -> str | None:
    """Pull the first ``sessionID`` field out of the NDJSON stream."""
    for obj in _iter_events(stdout):
        sid = obj.get("sessionID")
        if isinstance(sid, str) and sid:
            return sid
        # Some events nest it under ``part``.
        part = obj.get("part")
        if isinstance(part, dict):
            sid = part.get("sessionID")
            if isinstance(sid, str) and sid:
                return sid
    return None


def parse_tool_errors(stdout: str) -> list[str]:
    """Extract tool-call errors from opencode's NDJSON stream.

    Each ``tool_use`` event carries a ``part.state`` block with a
    ``status`` field (``"completed"`` / ``"error"``) and, on error,
    an ``error`` message + the failing ``input``. We surface every
    error as ``"<tool>: <message>"`` so the caller can fail loud
    rather than mistake a destructive or no-op tool call for a real
    edit.
    """
    errors: list[str] = []
    for obj in _iter_events(stdout):
        if obj.get("type") != "tool_use":
            continue
        part = obj.get("part") or {}
        state = part.get("state") or {}
        if state.get("status") != "error":
            continue
        tool = part.get("tool") or "<unknown>"
        msg = state.get("error") or "(no error message)"
        errors.append(f"{tool}: {msg}")
    return errors


# Retained for tests and back-compat callers. Not used by OpenCodeDriver
# at runtime — the equivalent JSON is baked into the per-agent image.
_MINIMAL_AGENT_PROMPT = (
    "You are a coding assistant. Always make code changes by CALLING "
    "the edit tool — do NOT just describe the change in prose. The "
    "user's task is incomplete until your tool call actually modifies "
    "the file. Use read first to see the current contents, then edit "
    "to change them. Stop after a successful edit."
)


def build_opencode_config(
    *,
    provider_name: str,
    base_url: str,
    model_id: str,
    context_window: int = 65536,
    max_tokens: int = 8192,
    minimal_agent: bool = False,
) -> dict:
    """Render an ``opencode.json`` payload. Kept for tests; the
    production path bakes the equivalent into the image."""
    cfg: dict = {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            provider_name: {
                "npm": "@ai-sdk/openai-compatible",
                "name": f"Local via agentcap proxy ({base_url})",
                "options": {"baseURL": base_url},
                "models": {
                    model_id: {
                        "name": model_id,
                        "options": {"max_tokens": max_tokens},
                        "limit": {"context": context_window, "output": max_tokens},
                    }
                },
            }
        },
        "model": f"{provider_name}/{model_id}",
    }
    if minimal_agent:
        cfg["agent"] = {
            "minimal": {
                "description": "Stripped agent for CI / small-model CPU runs.",
                "model": f"{provider_name}/{model_id}",
                "prompt": _MINIMAL_AGENT_PROMPT,
                "permission": {"*": "deny", "read": "allow", "edit": "allow"},
            }
        }
    return cfg


class OpenCodeDriver(AgentDriver):
    name = "opencode"

    def __init__(
        self,
        *,
        sandbox: Sandbox,
        binary: str = "opencode",
        model: str | None = None,
        cwd: Path | str | None = None,
        provider_name: str = _DEFAULT_PROVIDER_NAME,
        extra_args: Sequence[str] = (),
        minimal_agent: bool = False,
    ) -> None:
        self.sandbox = sandbox
        self.binary = binary
        self.model = model
        self.cwd = str(cwd) if cwd is not None else None
        self.provider_name = provider_name
        self.extra_args = list(extra_args)
        self.minimal_agent = minimal_agent

    def close(self) -> None:
        """No-op. Per-run state lives in the buildah container's
        OverlayFS upper layer."""

    def _build_argv(
        self, prompt: str, *, session_id: str | None = None
    ) -> list[str]:
        argv = [self.binary, "run", "--format", "json"]
        if self.model:
            argv.extend(["--model", f"{self.provider_name}/{self.model}"])
        if self.minimal_agent:
            argv.extend(["--agent", "minimal"])
        if session_id:
            argv.extend(["--session", session_id])
        argv.extend(self.extra_args)
        argv.append(prompt)
        return argv

    def _run(
        self,
        argv: list[str],
        env: dict | None,
        timeout: float | None,
    ) -> subprocess.CompletedProcess:
        return self.sandbox.run(
            argv,
            env=env or {},
            cwd=self.cwd,
            timeout=timeout,
        )

    def start(
        self,
        prompt: str,
        *,
        env: dict | None = None,
        timeout: float | None = None,
    ) -> AgentTurn:
        proc = self._run(self._build_argv(prompt), env, timeout)
        return AgentTurn(
            session_id=parse_session_id(proc.stdout),
            response_text=parse_response_text(proc.stdout),
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            tool_errors=parse_tool_errors(proc.stdout),
        )

    def resume(
        self,
        prompt: str,
        *,
        session_id: str,
        env: dict | None = None,
        timeout: float | None = None,
    ) -> AgentTurn:
        proc = self._run(
            self._build_argv(prompt, session_id=session_id), env, timeout
        )
        return AgentTurn(
            session_id=parse_session_id(proc.stdout) or session_id,
            response_text=parse_response_text(proc.stdout),
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            tool_errors=parse_tool_errors(proc.stdout),
        )
