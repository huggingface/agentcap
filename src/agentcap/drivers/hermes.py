"""Hermes driver.

Drives ``hermes chat -q "<prompt>"`` non-interactively. ``~/.hermes/``
is baked into the per-agent image with the proxy URL and context
length pointing at the in-process proxy — see
[containers/agentcap-hermes.Containerfile](
../../../containers/agentcap-hermes.Containerfile). The driver does
no per-run config rewriting.

Identity content (``SOUL.md``, etc.) and per-run state (``memories/``,
``sessions/``, ``logs/``) all live under the image's ``/root/.hermes/``;
writes from the agent go to the buildah container's OverlayFS upper
layer and are discarded when the sandbox closes. Session
continuity across turns within one ``agentcap run`` falls out of the
container's persistence.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Sequence

import yaml

from . import AgentDriver, AgentTurn
from ..sandbox import Sandbox


_SESSION_ID_RE = re.compile(r"session_id:\s*([a-zA-Z0-9_\-]+)")
_RESUMED_MARKER = "Resumed"


def parse_session_id(output: str) -> str | None:
    m = _SESSION_ID_RE.search(output)
    return m.group(1) if m else None


def parse_tool_errors(stdout: str) -> list[str]:
    # TODO: hermes' tool-error format is not yet characterised.
    return []


def parse_response_text(stdout: str) -> str:
    """Extract the assistant body from a hermes run.

    For a resumed session, hermes prints a ``↻ Resumed <id>`` marker
    before the new turn — we slice everything after the last such
    marker. For an initial run we use the whole stdout. Then strip
    bare ``session_id:`` lines and surrounding whitespace.
    """
    lines = stdout.splitlines()
    last = -1
    for i, line in enumerate(lines):
        if _RESUMED_MARKER in line and "↻" in line:
            last = i
    body_lines = lines[last + 1 :] if last >= 0 else lines
    cleaned = [
        l for l in body_lines if not _SESSION_ID_RE.match(l.strip())
    ]
    return "\n".join(cleaned).strip()


def _rewrite_config(
    config_text: str,
    *,
    base_url: str,
    context_length_override: int | None = None,
) -> str:
    """Round-trip a hermes ``config.yaml`` through PyYAML, overriding
    ``model.base_url`` and (optionally) ``context_length``. Kept for
    unit tests; the production path bakes the equivalent into the
    image, so the driver never calls this at runtime."""
    cfg = yaml.safe_load(config_text) or {}
    if not isinstance(cfg, dict):
        raise ValueError("hermes config.yaml is not a YAML mapping")

    model = cfg.setdefault("model", {})
    if not isinstance(model, dict):
        raise ValueError("hermes config.yaml: 'model' must be a mapping")
    model["base_url"] = base_url

    if context_length_override is not None:
        model["context_length"] = context_length_override
        aux = cfg.setdefault("auxiliary", {})
        if not isinstance(aux, dict):
            raise ValueError(
                "hermes config.yaml: 'auxiliary' must be a mapping"
            )
        comp = aux.setdefault("compression", {})
        if not isinstance(comp, dict):
            raise ValueError(
                "hermes config.yaml: 'auxiliary.compression' must be a mapping"
            )
        comp["context_length"] = context_length_override

    return yaml.safe_dump(cfg, sort_keys=False)


class HermesDriver(AgentDriver):
    name = "hermes"

    def __init__(
        self,
        *,
        sandbox: Sandbox,
        binary: str = "hermes",
        model: str | None = None,
        extra_args: Sequence[str] = ("-Q", "--yolo", "--accept-hooks"),
        cwd: Path | str | None = None,
        ignore_rules: bool = False,
        toolsets: str | None = None,
    ) -> None:
        # cwd: sandbox-side working directory. Hermes auto-injects
        # AGENTS.md / CLAUDE.md / .cursorrules from its cwd into every
        # system prompt; the orchestrator typically passes the result
        # of ``sandbox.mkdtemp`` so per-run cwd state doesn't leak.
        #
        # ignore_rules / toolsets shrink the default Hermes system
        # prompt for CPU + small-model runs.
        #
        # model: passed via ``hermes chat -m <id>``. The CLI flag is
        # the only path that reliably populates the ``model`` field
        # in the outbound OAI request body; ``model.name`` in
        # ``config.yaml`` doesn't propagate for every provider profile.
        self.sandbox = sandbox
        self.binary = binary
        self.model = model
        self.extra_args = list(extra_args)
        self.cwd = str(cwd) if cwd is not None else None
        self.ignore_rules = ignore_rules
        self.toolsets = toolsets

    def close(self) -> None:
        """No-op. Per-run state lives in the buildah container's
        OverlayFS upper layer."""

    def _build_argv(
        self, prompt: str, *, session_id: str | None
    ) -> list[str]:
        argv = [self.binary, "chat", "-q", prompt, *self.extra_args]
        if self.model:
            argv.extend(["-m", self.model])
        if self.ignore_rules:
            argv.append("--ignore-rules")
        if self.toolsets:
            argv.extend(["-t", self.toolsets])
        if session_id is None:
            argv.append("--pass-session-id")
        else:
            argv.extend(["--resume", session_id])
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
        proc = self._run(
            self._build_argv(prompt, session_id=None), env, timeout
        )
        combined = proc.stdout + "\n" + proc.stderr
        return AgentTurn(
            session_id=parse_session_id(combined),
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
            session_id=session_id,
            response_text=parse_response_text(proc.stdout),
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            tool_errors=parse_tool_errors(proc.stdout),
        )
