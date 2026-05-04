"""Goose driver.

Drives ``goose run -t "<prompt>"`` non-interactively. Goose uses the
``OPENAI_HOST`` / ``OPENAI_API_KEY`` env vars (plus ``GOOSE_PROVIDER``
and ``GOOSE_MODEL``) to find a model when the openai-compat provider
is selected — ``OPENAI_HOST`` should be the **host root**
(``http://127.0.0.1:8001`` — Goose appends its own ``/v1/...`` path).

Native sessions: ``--name <name>`` writes a session under the goose
config dir; ``--resume`` continues it. The driver invents a stable
session name on ``start`` and reuses it on ``resume``. To keep
session storage out of the user's ``~/.config/goose``, the driver
points ``XDG_CONFIG_HOME`` at a temp overlay; ``close`` discards it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Sequence

from . import AgentDriver, AgentTurn


class GooseDriver(AgentDriver):
    name = "goose"

    def __init__(
        self,
        binary: str = "goose",
        model: str | None = None,
        proxy_base_url: str | None = None,
        api_key: str = "dummy",
        cwd: Path | str | None = None,
        extra_args: Sequence[str] = (),
    ) -> None:
        self.binary = binary
        self.model = model
        # Goose wants the host root, not a /v1 endpoint — strip if the
        # caller passed an OpenAI-style /v1 base URL.
        self.proxy_host = (
            proxy_base_url.rstrip("/").removesuffix("/v1")
            if proxy_base_url
            else None
        )
        self.api_key = api_key
        self.cwd = Path(cwd) if cwd is not None else None
        self.extra_args = list(extra_args)
        self._overlay_config: Path | None = None

    def _ensure_overlay(self) -> Path:
        """Create a temp XDG_CONFIG_HOME so goose state never lands in
        the user's ``~/.config/goose``."""
        if self._overlay_config is None:
            self._overlay_config = Path(
                tempfile.mkdtemp(prefix="agentcap-goose-")
            )
            (self._overlay_config / "goose").mkdir(parents=True, exist_ok=True)
        return self._overlay_config

    def close(self) -> None:
        if self._overlay_config is not None and self._overlay_config.is_dir():
            shutil.rmtree(self._overlay_config, ignore_errors=True)
            self._overlay_config = None

    def _build_argv(
        self, prompt: str, *, session_name: str | None, resume: bool
    ) -> list[str]:
        argv = [self.binary, "run", "-t", prompt, *self.extra_args]
        if session_name is None:
            argv.append("--no-session")
        else:
            argv.extend(["--name", session_name])
            if resume:
                argv.append("--resume")
        return argv

    def _run(
        self,
        argv: list[str],
        env: dict | None,
        timeout: float | None,
    ) -> subprocess.CompletedProcess:
        full_env = {**os.environ}
        full_env["XDG_CONFIG_HOME"] = str(self._ensure_overlay())
        if self.proxy_host is not None:
            full_env["OPENAI_HOST"] = self.proxy_host
        if self.model:
            full_env["GOOSE_PROVIDER"] = full_env.get("GOOSE_PROVIDER", "openai")
            full_env["GOOSE_MODEL"] = self.model
        full_env.setdefault("OPENAI_API_KEY", self.api_key)
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
        # Pre-mint a session name so resume() can reuse it. Callers
        # that want ephemeral runs can pass ``no_session=True`` via
        # ``env`` — but the simpler path is just to ignore the
        # returned session_id.
        session_name = f"agentcap-{uuid.uuid4().hex[:8]}"
        proc = self._run(
            self._build_argv(prompt, session_name=session_name, resume=False),
            env,
            timeout,
        )
        return AgentTurn(
            session_id=session_name,
            response_text=proc.stdout.strip(),
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
            self._build_argv(prompt, session_name=session_id, resume=True),
            env,
            timeout,
        )
        return AgentTurn(
            session_id=session_id,
            response_text=proc.stdout.strip(),
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
        )
