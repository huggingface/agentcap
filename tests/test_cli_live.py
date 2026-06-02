"""End-to-end live test for ``agentcap run``.

Exercises the full CLI → orchestrator → sandbox → real agent path
against a real OpenAI-compat ``/v1`` server (set
``AGENTCAP_TEST_LLM_URL`` or have ``llama`` on PATH so the fixture
runs ``llama serve``). Replaces the heavily-mocked plumbing tests
previously in ``test_cli.py``:
``test_run_synthesized_defaults_from_upstream_and_model`` and
``test_run_invokes_orchestrator_under_proxy``.

Pi is the agent under test — its image install is small, sessions
stream as per-file JSONL through the symlink (no SQLite dump
required), and it's the most CI-friendly of the four agents.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from agentcap.__main__ import cli


@pytest.mark.live
def test_agentcap_run_live(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    live_proxy_base_url: str,
    live_model: str,
    agentcap_image_for,
):
    """``agentcap run --agent pi`` against a real model server.

    Verifies the CLI plumbing the mocked tests used to cover:
    - flag parsing → ``AGENTCAP_PROXY_URL`` / ``AGENTCAP_MODEL`` /
      ``AGENTCAP_PROVIDER`` / ``AGENTCAP_TRACES_DIR`` /
      ``AGENTCAP_STATE_DIR`` reach the sandbox,
    - the in-process proxy wraps the orchestrator (captures land in
      ``<run_dir>/captures/``),
    - per-run ``traces/`` is populated as the agent runs (pi streams
      JSONL through the in-container symlink),
    - ``run.json`` summary is written with the right shape.

    No internal monkeypatching — the only env manipulation is
    ``AGENTCAP_WORKSPACE`` (a legitimate CLI input).
    """
    # Pre-build the pi image. The fixture is also pulled in by the
    # sandbox-using live tests; first call builds, subsequent calls
    # are a no-op.
    agentcap_image_for("pi")

    tasks = tmp_path / "tasks.txt"
    tasks.write_text("Say hello in one short sentence, then stop.\n")

    workspace = tmp_path / "ws"
    workspace.mkdir()
    monkeypatch.setenv("AGENTCAP_WORKSPACE", str(workspace))

    # ``live_proxy_base_url`` ends with ``/v1`` (agent-side path);
    # ``--upstream`` wants the server root.
    upstream = live_proxy_base_url
    if upstream.endswith("/v1"):
        upstream = upstream[: -len("/v1")]

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "run",
            "--agent", "pi",
            "--model", live_model,
            "--upstream", upstream,
            "--tasks", str(tasks),
            "--turns", "1",
            "--timeout", "180",
        ],
    )
    assert result.exit_code == 0, result.output

    # One run dir was created under the workspace.
    run_dirs = sorted((workspace / ".agentcap").glob("pi-*"))
    assert len(run_dirs) == 1, run_dirs
    run_dir = run_dirs[0]

    # run.json shape — same assertions the mocked predecessor made.
    summary = json.loads((run_dir / "run.json").read_text())
    assert summary["agent"] == "pi"
    assert summary["model"] == live_model
    assert summary["upstream"] == upstream
    assert summary["turns_per_task"] == 1
    assert len(summary["tasks"]) == 1
    task = summary["tasks"][0]
    assert task["completed_turns"] == 1
    assert task["session_id"], "pi should mint a session id"

    # Captures landed on disk via the in-process proxy.
    captures = list((run_dir / "captures").glob("*.request.json"))
    assert captures, "proxy should have captured at least one request"

    # Pi's native session JSONL landed via the in-container symlink.
    traces = list((run_dir / "traces").iterdir())
    assert traces, "pi should have streamed at least one trace file"
    assert any(f.suffix == ".jsonl" for f in traces)
