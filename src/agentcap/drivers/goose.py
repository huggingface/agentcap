"""Goose driver.

Drives ``goose run -t "<prompt>"`` non-interactively. The proxy URL +
provider + ``OPENAI_API_KEY`` are baked into the per-agent image's
ENV (see [containers/agentcap-goose.Containerfile](
../../../containers/agentcap-goose.Containerfile)); the driver only
sets ``GOOSE_MODEL`` per run.

Goose's own session state lives at ``~/.config/goose/sessions/``
inside the sandbox. The buildah working container's OverlayFS upper
layer captures those writes — they survive between turns within one
``agentcap run`` and are discarded when the sandbox closes.
"""

from __future__ import annotations

import subprocess
import uuid
from pathlib import Path
from typing import Sequence

from . import AgentDriver, AgentTurn
from ..sandbox import Sandbox


def parse_tool_errors(stdout: str) -> list[str]:
    # TODO: goose's tool-error format is not yet characterised.
    return []


class GooseDriver(AgentDriver):
    name = "goose"

    def __init__(
        self,
        *,
        sandbox: Sandbox,
        binary: str = "goose",
        model: str | None = None,
        cwd: Path | str | None = None,
        extra_args: Sequence[str] = (),
    ) -> None:
        self.sandbox = sandbox
        self.binary = binary
        self.model = model
        # ``cwd`` is sandbox-side; on bwrap a host path bound into
        # the sandbox by BwrapSandbox.
        self.cwd = str(cwd) if cwd is not None else None
        self.extra_args = list(extra_args)

    def close(self) -> None:
        """No-op. Session state lives in the buildah container's
        OverlayFS upper layer, which the sandbox tears down when
        it's closed."""

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
        full_env: dict[str, str] = {}
        if self.model:
            full_env["GOOSE_MODEL"] = self.model
        if env:
            full_env.update(env)
        return self.sandbox.run(
            argv,
            env=full_env,
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
            tool_errors=parse_tool_errors(proc.stdout),
        )
