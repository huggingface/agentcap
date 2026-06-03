"""Live integration tests for each agent driver.

Verifies the infrastructure path only: agent runs inside its per-agent
podman container, dials the in-process capture proxy, and gets a
response back. Agent output *quality* — whether the model emits a
syntactically valid tool call, whether it picks the right file, etc.
— is intentionally not asserted. A separate (model-grading) test
would be the place for that.

Assertions per agent: ``returncode == 0``, ``turn.tool_errors`` empty,
``turn.response_text`` non-empty (the agent received at least one
model response through the proxy).
"""

from __future__ import annotations

import pytest

from agentcap.drivers.goose import GooseDriver
from agentcap.drivers.hermes import HermesDriver
from agentcap.drivers.opencode import OpenCodeDriver
from agentcap.drivers.pi import PiDriver


INFRA_PROMPT = "Say hi, then stop."


def _assert_infrastructure_works(turn) -> None:
    assert turn.returncode == 0, (
        f"agent exited rc={turn.returncode}\n"
        f"--- stdout (tail) ---\n{turn.stdout[-500:]}\n"
        f"--- stderr (tail) ---\n{turn.stderr[-500:]}"
    )
    assert not turn.tool_errors, (
        f"{len(turn.tool_errors)} tool-call error(s):\n"
        + "\n".join(f"  - {e}" for e in turn.tool_errors)
    )
    assert turn.response_text, (
        f"agent produced no response text — wire path may be broken.\n"
        f"--- stdout (tail) ---\n{turn.stdout[-500:]}\n"
        f"--- stderr (tail) ---\n{turn.stderr[-500:]}"
    )


@pytest.mark.live
def test_goose_live(live_proxy_base_url, live_model, agent_proj_for):
    sandbox, proj = agent_proj_for("goose")
    drv = GooseDriver(
        sandbox=sandbox, binary="goose", model=live_model, cwd=proj,
    )
    try:
        turn = drv.start(INFRA_PROMPT, timeout=900)
        assert turn.session_id and turn.session_id.startswith("agentcap-")
        _assert_infrastructure_works(turn)
    finally:
        drv.close()


@pytest.mark.live
def test_pi_live(live_proxy_base_url, live_model, agent_proj_for):
    sandbox, proj = agent_proj_for("pi")
    drv = PiDriver(
        sandbox=sandbox, binary="pi", model=live_model, cwd=proj,
    )
    try:
        turn = drv.start(INFRA_PROMPT, timeout=900)
        _assert_infrastructure_works(turn)
    finally:
        drv.close()


@pytest.mark.live
@pytest.mark.skip(
    reason=(
        "opencode 1.15.x doesn't pick up the baked ``agent.minimal`` from "
        "``~/.config/opencode/opencode.json`` inside the per-agent "
        "container — fails with ``agent \"minimal\" not found`` and "
        "``Model not found`` even with ``mode: primary`` + explicit "
        "model. Needs investigation: instrument the init script with "
        "``opencode debug config`` to see what config opencode actually "
        "resolves."
    )
)
def test_opencode_live(live_proxy_base_url, live_model, agent_proj_for):
    sandbox, proj = agent_proj_for("opencode")
    # OpenCode recursively globs from / in empty dirs; seed a
    # package.json to bound its exploration.
    sandbox.write_text(
        f"{proj}/package.json", '{"name":"smoke","version":"0.0.0"}\n'
    )
    drv = OpenCodeDriver(
        sandbox=sandbox, binary="opencode", model=live_model, cwd=proj,
        minimal_agent=True,
    )
    try:
        turn = drv.start(INFRA_PROMPT, timeout=900)
        _assert_infrastructure_works(turn)
    finally:
        drv.close()


@pytest.mark.live
def test_hermes_live(live_proxy_base_url, agent_proj_for):
    sandbox, proj = agent_proj_for("hermes")
    drv = HermesDriver(
        sandbox=sandbox, binary="hermes", cwd=proj,
        ignore_rules=True, toolsets="file",
    )
    try:
        turn = drv.start(INFRA_PROMPT, timeout=900)
        assert turn.session_id is not None
        _assert_infrastructure_works(turn)
    finally:
        drv.close()
