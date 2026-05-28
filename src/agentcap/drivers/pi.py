"""pi-mono coding-agent driver.

Drives ``pi -p "<prompt>" --provider local --model <id>`` non-
interactively. The provider config (proxy URL, model entries) and
PI_CODING_AGENT_DIR are baked into the per-agent image — see
[containers/agentcap-pi.Containerfile](
../../../containers/agentcap-pi.Containerfile). The driver passes
the model id at the CLI.

Native sessions: pi tracks the most recent session under
``PI_CODING_AGENT_SESSION_DIR`` and resumes via ``--continue``. The
driver lets pi mint its own UUID on ``start`` (no flag), then passes
``--continue`` on ``resume``. Session state lives in the buildah
container's OverlayFS upper layer — survives across turns within
one ``agentcap run``, discarded when the sandbox closes.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Sequence

from . import AgentDriver, AgentTurn
from ..sandbox import Sandbox


_DEFAULT_PROVIDER_NAME = "local"


def parse_tool_errors(stdout: str) -> list[str]:
    # TODO: pi's tool-error format is not yet characterised.
    return []


def build_models_json(
    *,
    provider_name: str,
    base_url: str,
    model_id: str,
    api_key_env: str = "PI_LOCAL_API_KEY",
    context_window: int = 65536,
    max_tokens: int = 4096,
) -> dict:
    """Render a pi ``models.json`` payload. Kept for tests; the
    production path bakes the equivalent into the image."""
    return {
        "providers": {
            provider_name: {
                "baseUrl": base_url,
                "api": "openai-completions",
                "apiKey": api_key_env,
                "compat": {
                    "supportsDeveloperRole": False,
                    "supportsReasoningEffort": False,
                },
                "models": [
                    {
                        "id": model_id,
                        "name": model_id,
                        "reasoning": False,
                        "input": ["text"],
                        "contextWindow": context_window,
                        "maxTokens": max_tokens,
                        "cost": {
                            "input": 0, "output": 0,
                            "cacheRead": 0, "cacheWrite": 0,
                        },
                    }
                ],
            }
        }
    }


class PiDriver(AgentDriver):
    name = "pi"

    def __init__(
        self,
        *,
        sandbox: Sandbox,
        binary: str = "pi",
        model: str | None = None,
        cwd: Path | str | None = None,
        provider_name: str = _DEFAULT_PROVIDER_NAME,
        extra_args: Sequence[str] = (),
    ) -> None:
        self.sandbox = sandbox
        self.binary = binary
        self.model = model
        self.cwd = str(cwd) if cwd is not None else None
        self.provider_name = provider_name
        self.extra_args = list(extra_args)

    def close(self) -> None:
        """No-op. Per-run state lives in the buildah container's
        OverlayFS upper layer."""

    def _build_argv(
        self,
        prompt: str,
        *,
        resume: bool,
        no_session: bool,
    ) -> list[str]:
        argv = [
            self.binary,
            "-p",
            prompt,
            "--provider",
            self.provider_name,
            *self.extra_args,
        ]
        if self.model:
            argv.extend(["--model", self.model])
        if no_session:
            argv.append("--no-session")
        elif resume:
            argv.append("--continue")
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
        # No --session: pi mints its own UUID and writes it under
        # PI_CODING_AGENT_SESSION_DIR. Resume picks the latest via
        # --continue (synthetic marker returned to the orchestrator).
        proc = self._run(
            self._build_argv(prompt, resume=False, no_session=False),
            env,
            timeout,
        )
        return AgentTurn(
            session_id="latest",
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
            self._build_argv(prompt, resume=True, no_session=False),
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
