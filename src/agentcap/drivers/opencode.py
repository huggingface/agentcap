"""OpenCode driver.

OpenCode emits NDJSON events on stdout when invoked with
``--format json``. ``text`` events carry assistant chunks; the
session id appears in every event as ``sessionID``. Multi-turn is
via ``opencode run --session <id>``: the driver parses the id from
``start`` output and reuses it on ``resume``.

The model endpoint is configured via an ``opencode.json`` written
into ``cwd`` when ``proxy_base_url`` is set. Always launch from a
real project dir — opencode hangs ≥30 min if the model directs it
to recursively glob from filesystem root.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Sequence

from . import AgentDriver, AgentTurn


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


_MINIMAL_AGENT_PROMPT = (
    "You are a coding assistant. Always make code changes by CALLING "
    "the edit tool — do NOT just describe the change in prose. The "
    "user's task is incomplete until your tool call actually modifies "
    "the file. Use read first to see the current contents, then edit "
    "to change them. Stop after a successful edit."
)

# Minimal-agent permissions: deny all tools then explicitly allow
# read + edit. opencode resolves rules via findLast, so the specific
# allows override the wildcard deny. Use the ``permission`` field
# (not the deprecated ``tools`` field, which has surprising aliasing
# of write/multiedit/patch onto ``permission.edit``).
_MINIMAL_AGENT_PERMISSION: dict[str, str] = {
    "*": "deny",
    "read": "allow",
    "edit": "allow",
}


def build_opencode_config(
    *,
    provider_name: str,
    base_url: str,
    model_id: str,
    context_window: int = 65536,
    max_tokens: int = 8192,
    minimal_agent: bool = False,
) -> dict:
    """Render an ``opencode.json`` payload that wires a local
    OpenAI-compatible provider at ``base_url``.

    With ``minimal_agent=True`` the config also defines an
    ``agent.minimal`` entry with a stripped system prompt and only
    ``read`` + ``edit`` tools enabled. Pass ``--agent minimal`` (or
    ``OpenCodeDriver(minimal_agent=True)`` so the driver does it) to
    use it. Trades fidelity for speed on tool-heavy CPU runs.
    """
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
                "permission": dict(_MINIMAL_AGENT_PERMISSION),
            }
        }
    return cfg


class OpenCodeDriver(AgentDriver):
    name = "opencode"

    def __init__(
        self,
        binary: str = "opencode",
        model: str | None = None,
        proxy_base_url: str | None = None,
        cwd: Path | str | None = None,
        provider_name: str = _DEFAULT_PROVIDER_NAME,
        context_window: int = 65536,
        max_tokens: int = 8192,
        extra_args: Sequence[str] = (),
        minimal_agent: bool = False,
    ) -> None:
        # minimal_agent: write a stripped-down ``agent.minimal`` into
        # the generated opencode.json (system prompt + only read/edit
        # tools) and invoke ``opencode run --agent minimal``. Off by
        # default — capture runs want the realistic surface.
        self.binary = binary
        self.model = model
        self.proxy_base_url = proxy_base_url
        self.cwd = Path(cwd) if cwd is not None else None
        self.provider_name = provider_name
        self.context_window = context_window
        self.max_tokens = max_tokens
        self.extra_args = list(extra_args)
        self.minimal_agent = minimal_agent
        self._wrote_config: Path | None = None

    def _maybe_write_config(self) -> None:
        """Write opencode.json into ``cwd`` if proxy redirect requested.

        Skipped when ``cwd`` is ``None`` (caller is responsible for
        providing config) or when no ``proxy_base_url`` was given.
        """
        if self.proxy_base_url is None or self.model is None or self.cwd is None:
            return
        if self._wrote_config is not None:
            return
        cfg_path = self.cwd / "opencode.json"
        cfg_path.write_text(
            json.dumps(
                build_opencode_config(
                    provider_name=self.provider_name,
                    base_url=self.proxy_base_url,
                    model_id=self.model,
                    context_window=self.context_window,
                    max_tokens=self.max_tokens,
                    minimal_agent=self.minimal_agent,
                ),
                indent=2,
            )
        )
        self._wrote_config = cfg_path

    def close(self) -> None:
        if self._wrote_config is not None and self._wrote_config.is_file():
            try:
                self._wrote_config.unlink()
            except OSError:
                pass
            self._wrote_config = None

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
        full_env = {**os.environ, "OPENAI_API_KEY": "dummy"}
        if env:
            full_env.update(env)
        return subprocess.run(
            argv,
            env=full_env,
            cwd=str(self.cwd) if self.cwd is not None else None,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def start(
        self,
        prompt: str,
        *,
        env: dict | None = None,
        timeout: float | None = None,
    ) -> AgentTurn:
        self._maybe_write_config()
        proc = self._run(self._build_argv(prompt), env, timeout)
        return AgentTurn(
            session_id=parse_session_id(proc.stdout),
            response_text=parse_response_text(proc.stdout),
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
        )

    def resume(
        self,
        prompt: str,
        *,
        session_id: str,
        env: dict | None = None,
        timeout: float | None = None,
    ) -> AgentTurn:
        self._maybe_write_config()
        proc = self._run(
            self._build_argv(prompt, session_id=session_id), env, timeout
        )
        return AgentTurn(
            session_id=parse_session_id(proc.stdout) or session_id,
            response_text=parse_response_text(proc.stdout),
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
        )
