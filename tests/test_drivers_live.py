"""Live integration tests for each agent driver.

Each test instantiates the driver against a real model server and
asks the agent to add a docstring to a tiny ``hello.py``. Skips
unless the agent binary and a model endpoint are configured (see
README ``Running tests``).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agentcap.drivers.goose import GooseDriver
from agentcap.drivers.hermes import HermesDriver
from agentcap.drivers.opencode import OpenCodeDriver
from agentcap.drivers.pi import PiDriver

from .conftest import DOCSTRING_PROMPT


def _docstring_landed(proj: Path) -> bool:
    return '"""' in (proj / "hello.py").read_text()


# Reset between retries — a previous attempt may have left a partial
# edit. Small-model sampling occasionally drops the tool call.
_HELLO_PY = 'def hello():\n    print("Hello, world!")\n'


def _run_with_retry(
    drv, prompt: str, proj: Path, *, timeout: float, retries: int = 3
):
    last_turn = None
    for attempt in range(1, retries + 1):
        (proj / "hello.py").write_text(_HELLO_PY)
        last_turn = drv.start(prompt, timeout=timeout)
        if _docstring_landed(proj) and last_turn.returncode == 0:
            return last_turn
    return last_turn


def _assert_docstring_added(proj: Path, turn) -> None:
    body = (proj / "hello.py").read_text()
    assert turn.returncode == 0, (
        f"agent exited rc={turn.returncode}\n"
        f"--- stdout (tail) ---\n{turn.stdout[-500:]}\n"
        f"--- stderr (tail) ---\n{turn.stderr[-500:]}"
    )
    assert '"""' in body, (
        f"docstring not added after retries; hello.py is:\n{body}\n"
        f"--- stdout (tail) ---\n{turn.stdout[-500:]}"
    )


@pytest.mark.live
def test_goose_live(
    goose_bin, live_proxy_base_url, live_model, hello_proj
):
    drv = GooseDriver(
        binary=goose_bin,
        model=live_model,
        proxy_base_url=live_proxy_base_url,
        cwd=hello_proj,
    )
    try:
        turn = _run_with_retry(drv, DOCSTRING_PROMPT, hello_proj, timeout=180)
        # Goose mints a session name on start that resume can reuse.
        assert turn.session_id and turn.session_id.startswith("agentcap-")
        _assert_docstring_added(hello_proj, turn)
    finally:
        drv.close()


@pytest.mark.live
def test_pi_live(pi_bin, live_proxy_base_url, live_model, hello_proj):
    drv = PiDriver(
        binary=pi_bin,
        model=live_model,
        proxy_base_url=live_proxy_base_url,
        cwd=hello_proj,
    )
    try:
        turn = _run_with_retry(drv, DOCSTRING_PROMPT, hello_proj, timeout=180)
        _assert_docstring_added(hello_proj, turn)
    finally:
        drv.close()


@pytest.mark.live
def test_opencode_live(
    opencode_bin, live_proxy_base_url, live_model, hello_proj
):
    # OpenCode hangs in empty dirs (recursive glob from /); seed a
    # token package.json so the agent has bounded content to explore.
    (hello_proj / "package.json").write_text(
        '{"name":"smoke","version":"0.0.0"}\n'
    )
    drv = OpenCodeDriver(
        binary=opencode_bin,
        model=live_model,
        proxy_base_url=live_proxy_base_url,
        cwd=hello_proj,
        # Stripped agent (read+edit only) for fast CPU runs.
        minimal_agent=True,
    )
    try:
        turn = _run_with_retry(drv, DOCSTRING_PROMPT, hello_proj, timeout=240)
        _assert_docstring_added(hello_proj, turn)
    finally:
        drv.close()


@pytest.mark.live
def test_hermes_live(
    hermes_bin, live_proxy_base_url, hello_proj
):
    """Hermes resolves its model from ``~/.hermes/config.yaml``; the
    driver overlays a fresh HERMES_HOME with that config rewritten to
    point at ``proxy_base_url``."""
    drv = HermesDriver(
        binary=hermes_bin,
        proxy_base_url=live_proxy_base_url,
        cwd=hello_proj,
        # CPU + small-model trims: skip rule/skill auto-injection,
        # narrow toolset, and override the ≥64K context guard.
        ignore_rules=True,
        toolsets="file",
        context_length_override=65536,
    )
    try:
        turn = _run_with_retry(drv, DOCSTRING_PROMPT, hello_proj, timeout=240)
        assert turn.session_id is not None
        _assert_docstring_added(hello_proj, turn)
    finally:
        drv.close()
