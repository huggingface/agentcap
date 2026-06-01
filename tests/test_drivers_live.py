"""Live integration tests for each agent driver.

Verifies the infrastructure path — agent runs inside its per-agent
sandbox, reaches the model server, fires its edit tool, mutates a
file the test reads back via ``sandbox.read_text``. Three assertions:
agent exited 0, file changed, ``turn.tool_errors`` empty. Agent
output quality is explicitly not graded here.

Skips when ``live_proxy_base_url`` can't be assembled (no
``$AGENTCAP_TEST_GGUF`` / no ``llama`` on PATH) or the agent binary
isn't on the sandbox PATH.
"""

from __future__ import annotations

import sys
import time

import pytest

from agentcap.drivers.goose import GooseDriver
from agentcap.drivers.hermes import HermesDriver
from agentcap.drivers.opencode import OpenCodeDriver
from agentcap.drivers.pi import PiDriver

from .conftest import DOCSTRING_PROMPT, _HELLO_PY, reset_hello_py


def _log(msg: str) -> None:
    """Stderr progress line (visible with ``pytest -s``)."""
    sys.stderr.write(f"  [agentcap-test] {msg}\n")
    sys.stderr.flush()


def _turn_is_clean(turn, sandbox, proj: str) -> bool:
    if turn.returncode != 0:
        return False
    if turn.tool_errors:
        return False
    return sandbox.read_text(f"{proj}/hello.py") != _HELLO_PY


def _run_with_retry(
    drv, prompt: str, sandbox, proj: str,
    *,
    timeout: float, retries: int = 3,
):
    last_turn = None
    for attempt in range(1, retries + 1):
        _log(f"{drv.name} attempt {attempt}/{retries} (timeout={timeout}s)…")
        reset_hello_py(sandbox, proj)
        t0 = time.monotonic()
        last_turn = drv.start(prompt, timeout=timeout)
        elapsed = time.monotonic() - t0
        edited = sandbox.read_text(f"{proj}/hello.py") != _HELLO_PY
        _log(
            f"{drv.name} attempt {attempt}: {elapsed:.1f}s "
            f"rc={last_turn.returncode} edited={edited} "
            f"tool_errors={len(last_turn.tool_errors)}"
        )
        if _turn_is_clean(last_turn, sandbox, proj):
            return last_turn
    return last_turn


def _assert_infrastructure_works(sandbox, proj: str, turn) -> None:
    body = sandbox.read_text(f"{proj}/hello.py")
    assert turn.returncode == 0, (
        f"agent exited rc={turn.returncode}\n"
        f"--- stdout (tail) ---\n{turn.stdout[-500:]}\n"
        f"--- stderr (tail) ---\n{turn.stderr[-500:]}"
    )
    assert not turn.tool_errors, (
        f"{len(turn.tool_errors)} tool-call error(s):\n"
        + "\n".join(f"  - {e}" for e in turn.tool_errors)
        + f"\n--- stdout (tail) ---\n{turn.stdout[-500:]}"
    )
    assert body != _HELLO_PY, (
        f"agent did not edit hello.py after retries; file is still:\n"
        f"{body}\n--- stdout (tail) ---\n{turn.stdout[-500:]}"
    )


@pytest.mark.live
def test_goose_live(live_proxy_base_url, live_model, agent_proj_for):
    sandbox, proj = agent_proj_for("goose")
    drv = GooseDriver(
        sandbox=sandbox,
        binary="goose",
        model=live_model,
        cwd=proj,
    )
    try:
        turn = _run_with_retry(drv, DOCSTRING_PROMPT, sandbox, proj, timeout=300)
        assert turn.session_id and turn.session_id.startswith("agentcap-")
        _assert_infrastructure_works(sandbox, proj, turn)
    finally:
        drv.close()


@pytest.mark.live
def test_pi_live(live_proxy_base_url, live_model, agent_proj_for):
    sandbox, proj = agent_proj_for("pi")
    drv = PiDriver(
        sandbox=sandbox,
        binary="pi",
        model=live_model,
        cwd=proj,
    )
    try:
        turn = _run_with_retry(drv, DOCSTRING_PROMPT, sandbox, proj, timeout=300)
        _assert_infrastructure_works(sandbox, proj, turn)
    finally:
        drv.close()


@pytest.mark.live
def test_opencode_live(live_proxy_base_url, live_model, agent_proj_for):
    sandbox, proj = agent_proj_for("opencode")
    # OpenCode recursively globs from / in empty dirs; seed a
    # package.json to bound its exploration.
    sandbox.write_text(
        f"{proj}/package.json", '{"name":"smoke","version":"0.0.0"}\n'
    )
    drv = OpenCodeDriver(
        sandbox=sandbox,
        binary="opencode",
        model=live_model,
        cwd=proj,
        minimal_agent=True,
    )
    try:
        turn = _run_with_retry(drv, DOCSTRING_PROMPT, sandbox, proj, timeout=300)
        _assert_infrastructure_works(sandbox, proj, turn)
    finally:
        drv.close()


@pytest.mark.live
def test_hermes_live(live_proxy_base_url, agent_proj_for):
    sandbox, proj = agent_proj_for("hermes")
    drv = HermesDriver(
        sandbox=sandbox,
        binary="hermes",
        cwd=proj,
        # CPU + small-model trims.
        ignore_rules=True,
        toolsets="file",
    )
    try:
        turn = _run_with_retry(drv, DOCSTRING_PROMPT, sandbox, proj, timeout=300)
        assert turn.session_id is not None
        _assert_infrastructure_works(sandbox, proj, turn)
    finally:
        drv.close()
