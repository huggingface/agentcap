"""pi-mono coding-agent driver.

Drives ``pi -p "<prompt>" --provider local --model <id>`` non-
interactively. pi only accepts a custom OpenAI-compatible endpoint
through ``~/.pi/agent/models.json``, so the driver materialises a
sandboxed config dir on first use and points
``PI_CODING_AGENT_DIR`` at it.

llama.cpp's OpenAI shim doesn't accept the ``developer`` role pi uses
for reasoning-capable models, so we set
``compat.supportsDeveloperRole`` and ``compat.supportsReasoningEffort``
to ``false`` in the generated config.

Native sessions: pi tracks the most recent session under
``PI_CODING_AGENT_SESSION_DIR`` and resumes via ``--continue``. The
driver lets pi mint its own UUID on ``start`` (no flag), then passes
``--continue`` on ``resume``. Each driver instance gets its own
per-instance session dir so concurrent runs don't trample each
other. Telemetry / version-check are disabled via ``PI_OFFLINE`` +
``PI_SKIP_VERSION_CHECK``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence

from . import AgentDriver, AgentTurn


_DEFAULT_PROVIDER_NAME = "local"


def build_models_json(
    *,
    provider_name: str,
    base_url: str,
    model_id: str,
    api_key_env: str = "PI_LOCAL_API_KEY",
    context_window: int = 65536,
    max_tokens: int = 4096,
) -> dict:
    """Render a pi ``models.json`` payload that registers a single
    OpenAI-compatible local provider/model.

    ``api_key_env`` is the env-var name pi will read; pi requires an
    API key field even when the upstream ignores it.
    """
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
                            "input": 0,
                            "output": 0,
                            "cacheRead": 0,
                            "cacheWrite": 0,
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
        binary: str = "pi",
        model: str | None = None,
        proxy_base_url: str | None = None,
        api_key: str = "dummy",
        cwd: Path | str | None = None,
        provider_name: str = _DEFAULT_PROVIDER_NAME,
        context_window: int = 65536,
        max_tokens: int = 4096,
        extra_args: Sequence[str] = (),
    ) -> None:
        self.binary = binary
        self.model = model
        self.proxy_base_url = proxy_base_url
        self.api_key = api_key
        self.cwd = Path(cwd) if cwd is not None else None
        self.provider_name = provider_name
        self.context_window = context_window
        self.max_tokens = max_tokens
        self.extra_args = list(extra_args)
        self._overlay_dir: Path | None = None

    def _ensure_overlay(self) -> Path:
        """Build the PI_CODING_AGENT_DIR overlay on first use."""
        if self._overlay_dir is None:
            self._overlay_dir = Path(
                tempfile.mkdtemp(prefix="agentcap-pi-")
            )
            (self._overlay_dir / "sessions").mkdir(parents=True, exist_ok=True)
            if self.proxy_base_url is None or self.model is None:
                # Without redirect or model, leave models.json absent
                # and let pi fall through to its built-in providers.
                return self._overlay_dir
            payload = build_models_json(
                provider_name=self.provider_name,
                base_url=self.proxy_base_url,
                model_id=self.model,
                context_window=self.context_window,
                max_tokens=self.max_tokens,
            )
            (self._overlay_dir / "models.json").write_text(
                json.dumps(payload, indent=2)
            )
        return self._overlay_dir

    def close(self) -> None:
        if self._overlay_dir is not None and self._overlay_dir.is_dir():
            shutil.rmtree(self._overlay_dir, ignore_errors=True)
            self._overlay_dir = None

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
        full_env = {**os.environ}
        overlay = self._ensure_overlay()
        full_env["PI_CODING_AGENT_DIR"] = str(overlay)
        full_env["PI_CODING_AGENT_SESSION_DIR"] = str(overlay / "sessions")
        full_env.setdefault("PI_OFFLINE", "1")
        full_env.setdefault("PI_SKIP_VERSION_CHECK", "1")
        full_env["PI_LOCAL_API_KEY"] = self.api_key
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
        )
