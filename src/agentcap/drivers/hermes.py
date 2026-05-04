"""Hermes driver.

Wraps ``hermes chat -q "<prompt>"`` with the flags that the parent
project's working multi-turn shell script settled on:

  -Q --yolo --accept-hooks    # non-interactive, no confirmation prompts
  --pass-session-id           # initial turn only: print session id
  --resume <sid>              # continuation turns

When ``proxy_base_url`` is set, the driver builds a temporary
``HERMES_HOME`` whose ``config.yaml`` redirects the model endpoint at
the capture proxy. All other entries in the user's hermes home
(skills, sessions, state.db, etc.) are symlinked through, so the
agent's full configured behaviour is preserved. The user's real
``~/.hermes/config.yaml`` is never modified.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence

import yaml

from . import AgentDriver, AgentTurn


_SESSION_ID_RE = re.compile(r"session_id:\s*([a-zA-Z0-9_\-]+)")
_RESUMED_MARKER = "Resumed"


def parse_session_id(output: str) -> str | None:
    m = _SESSION_ID_RE.search(output)
    return m.group(1) if m else None


def parse_response_text(stdout: str) -> str:
    """Extract the assistant body from a hermes run.

    For a resumed session, hermes prints a ``↻ Resumed <id>`` marker
    before the new turn — we slice everything after the last such
    marker. For an initial run, we use the whole stdout. Then strip
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
    ``model.base_url`` and (optionally) the two context-length guards
    Hermes Agent enforces:

      - ``model.context_length``   (the chat model)
      - ``auxiliary.compression.context_length``  (the compression
        model — Hermes blocks startup if either is below 64 K)

    Other keys are preserved verbatim. Used by the overlay HERMES_HOME
    so a capture run can drive Hermes against a low-ctx model server
    (e.g. CPU + small GGUF in CI) without touching the user's real
    config.
    """
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


# Names under ~/.hermes/ that Hermes *writes to* during a run. The
# overlay creates these as fresh empty dirs (or absent files) so a
# capture run never reads or mutates the user's persistent state. Any
# `memory(action=add)` call by the agent lands here and gets discarded
# on driver close — instead of being written through the symlink into
# ~/.hermes/memories/MEMORY.md, where it would contaminate every
# future Hermes session AND silently shift the system-prompt prefix
# turn-over-turn within the capture run itself.
_HERMES_WRITABLE = frozenset({
    "memories",
    "sessions",
    "sandboxes",
    "state.db",
    "logs",
    "cron",
})


def build_overlay_hermes_home(
    proxy_base_url: str,
    *,
    user_hermes_home: Path | str = "~/.hermes",
    overlay_root: Path | str | None = None,
    context_length_override: int | None = None,
) -> Path:
    """Build a temporary HERMES_HOME that redirects to ``proxy_base_url``.

    The overlay is constructed so that:

    - ``config.yaml`` is regenerated from the user's copy with its
      ``base_url`` field swapped for ``proxy_base_url``.
    - Identity / read-only entries (``skills/``, ``SOUL.md``, …) are
      symlinked through to the user's home — the agent's behaviour is
      preserved verbatim.
    - Writable per-run state (``memories/``, ``sessions/``,
      ``state.db``, ``sandboxes/``, ``logs/``, ``cron/``) is created
      fresh in the overlay (empty dirs; ``state.db`` left absent for
      Hermes to recreate). This is a deliberate sandbox: agent writes
      stay inside the overlay and are discarded on driver close.

    Returns the path to the overlay directory; caller is responsible
    for cleanup (use ``shutil.rmtree`` once hermes has exited).
    """
    user = Path(os.path.expanduser(str(user_hermes_home))).resolve()
    if not user.is_dir():
        raise FileNotFoundError(f"hermes home not found: {user}")
    overlay = (
        Path(overlay_root)
        if overlay_root is not None
        else Path(tempfile.mkdtemp(prefix="agentcap-hermes-"))
    )
    overlay.mkdir(parents=True, exist_ok=True)

    user_cfg = user / "config.yaml"
    if not user_cfg.is_file():
        raise FileNotFoundError(f"user hermes config not found: {user_cfg}")

    for entry in user.iterdir():
        if entry.name == "config.yaml":
            continue
        target = overlay / entry.name
        if target.exists() or target.is_symlink():
            continue
        if entry.name in _HERMES_WRITABLE:
            # Fresh per-run state — empty dir if the user's copy is a
            # dir, otherwise leave absent so Hermes creates on demand.
            if entry.is_dir():
                target.mkdir()
            continue
        target.symlink_to(entry)

    rewritten = _rewrite_config(
        user_cfg.read_text(),
        base_url=proxy_base_url,
        context_length_override=context_length_override,
    )
    (overlay / "config.yaml").write_text(rewritten)
    return overlay


class HermesDriver(AgentDriver):
    name = "hermes"

    def __init__(
        self,
        binary: str = "hermes",
        extra_args: Sequence[str] = ("-Q", "--yolo", "--accept-hooks"),
        proxy_base_url: str | None = None,
        user_hermes_home: Path | str = "~/.hermes",
        cwd: Path | str | None = None,
        ignore_rules: bool = False,
        toolsets: str | None = None,
        context_length_override: int | None = None,
    ) -> None:
        # cwd: working directory for the hermes subprocess. Hermes
        # auto-injects AGENTS.md / CLAUDE.md / .cursorrules from its
        # cwd into the system prompt, so launching from agentcap's own
        # repo dir (or any project dir with those files) leaks those
        # files into every captured prompt. The orchestrator passes a
        # clean per-run sandbox here; tests can leave it None.
        #
        # ignore_rules / toolsets are knobs for shrinking Hermes'
        # system prompt — useful on CPU + small-model CI where the
        # default ~30 K-char system prompt (skills + identity +
        # full toolset) is the bottleneck. Off by default so capture
        # runs see the agent's realistic behaviour.
        self.binary = binary
        self.extra_args = list(extra_args)
        self.proxy_base_url = proxy_base_url
        self.user_hermes_home = user_hermes_home
        self.cwd = Path(cwd) if cwd is not None else None
        self.ignore_rules = ignore_rules
        self.toolsets = toolsets
        self.context_length_override = context_length_override
        self._overlay_home: Path | None = None

    def _ensure_overlay(self) -> Path | None:
        """Build the overlay HERMES_HOME on first use; return its path
        (or None when no proxy redirect was requested)."""
        if self.proxy_base_url is None:
            return None
        if self._overlay_home is None:
            self._overlay_home = build_overlay_hermes_home(
                self.proxy_base_url,
                user_hermes_home=self.user_hermes_home,
                context_length_override=self.context_length_override,
            )
        return self._overlay_home

    def close(self) -> None:
        """Remove the overlay HERMES_HOME if one was created."""
        if self._overlay_home is not None and self._overlay_home.is_dir():
            shutil.rmtree(self._overlay_home, ignore_errors=True)
            self._overlay_home = None

    def _build_argv(self, prompt: str, *, session_id: str | None) -> list[str]:
        argv = [self.binary, "chat", "-q", prompt, *self.extra_args]
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
        full_env = {**os.environ, "OPENAI_API_KEY": "dummy"}
        overlay = self._ensure_overlay()
        if overlay is not None:
            full_env["HERMES_HOME"] = str(overlay)
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
        )
