"""CLI smoke tests for `agentcap`.

These do not actually start a uvicorn server — they patch out
``agentcap.proxy.serve_in_thread`` and assert the right kwargs are
computed from the CLI flags. The proxy itself has its own integration
test suite.
"""

from __future__ import annotations

import os
import shutil
import types
from pathlib import Path

import pytest
from click.testing import CliRunner

from agentcap.__main__ import cli


def _has_trufflehog() -> bool:
    if shutil.which("trufflehog"):
        return True
    local = Path.home() / ".local" / "bin" / "trufflehog"
    return local.is_file() and os.access(local, os.X_OK)


_HAS_TRUFFLEHOG = _has_trufflehog()


@pytest.fixture(
    params=[
        pytest.param([], id="scan"),
        pytest.param(["--no-scan"], id="no-scan"),
    ]
)
def scan_args(request):
    """Yields ``[]`` (scan on, the default) or ``["--no-scan"]``.

    The scan-on variant requires trufflehog on PATH (or
    ~/.local/bin); without it, that parametrisation is skipped so
    the no-scan variant still runs."""
    if not request.param and not _HAS_TRUFFLEHOG:
        pytest.skip("trufflehog not installed; cannot exercise scan path")
    return request.param


def test_help_lists_subcommands():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    for sub in ("export", "run"):
        assert sub in result.output


def test_version_flag():
    from agentcap import __version__

    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_run_requires_agent_upstream_and_workdir():
    runner = CliRunner()
    result = runner.invoke(cli, ["run"])
    assert result.exit_code != 0
    # Click reports the first missing required option
    assert "--agent" in result.output


# Plumbing for ``agentcap run`` (CLI flag → env-var composition →
# orchestrator → run.json shape) is exercised end-to-end against a
# real model server in ``tests/test_cli_live.py::test_agentcap_run_live``.
# It replaces two previously heavily-mocked unit tests; the live test
# touches the real proxy + sandbox + agent so we don't have to stub
# them here.


def test_export_requires_push(tmp_path: Path):
    runner = CliRunner()
    result = runner.invoke(cli, ["export", str(tmp_path)])
    assert result.exit_code != 0
    assert "--push" in result.output


def test_export_requires_targets_or_all(tmp_path: Path):
    runner = CliRunner()
    result = runner.invoke(
        cli, ["export", "--push", "me/d"]
    )
    assert result.exit_code != 0
    assert "run-ids" in result.output or "--all" in result.output


def test_export_rejects_both_targets_and_all(tmp_path: Path):
    capture = tmp_path / "capture"
    capture.mkdir()
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["export", str(capture), "--all", "--push", "me/d"],
    )
    assert result.exit_code != 0
    assert "not both" in result.output


def test_run_hf_router_api_key_auto_from_hf_token_env(
    tmp_path: Path, monkeypatch, fake_sandbox
):
    import contextlib

    from agentcap.drivers import AgentTurn

    tasks = tmp_path / "tasks.txt"
    tasks.write_text("a task\n")

    class _FakeDriver:
        name = "hermes"

        def start(self, prompt, *, env=None, timeout=None):
            return AgentTurn(
                session_id="ses_xyz", response_text="r", returncode=0,
                stdout="", stderr="",
            )

        def resume(self, prompt, *, session_id, env=None, timeout=None):
            return AgentTurn(
                session_id=session_id, response_text="r", returncode=0,
                stdout="", stderr="",
            )

    monkeypatch.setattr(
        "agentcap.drivers.get_driver", lambda name, **kw: _FakeDriver()
    )
    monkeypatch.setattr(
        "agentcap.sandbox.require_sandbox_or_die",
        lambda **kw: fake_sandbox,
    )

    @contextlib.contextmanager
    def fake_proxy(*args, **kwargs):
        yield types.SimpleNamespace(host="127.0.0.1", port=18001)

    monkeypatch.setattr("agentcap.proxy.serve_in_thread", fake_proxy)
    monkeypatch.setenv("HF_TOKEN", "hf_env_token")
    monkeypatch.setenv("AGENTCAP_WORKSPACE", str(tmp_path))

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "run",
            "--agent", "hermes",
            "--model", "Qwen/Qwen3-8B",
            "--upstream", "https://router.huggingface.co",
            "--tasks", str(tasks),
            "--turns", "1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "HF Router token source=HF_TOKEN" in result.output


def _write_capture(capture_dir: Path, rid: str, model: str) -> None:
    import json
    (capture_dir / f"{rid}.request.json").write_text(json.dumps({
        "request_id": rid, "captured_at": 1,
        "body": {"model": model, "messages": []},
    }))


def test_export_auto_detects_model_from_captures(
    tmp_path: Path, fake_hf_api, scan_args,
):
    """The model auto-detected from captures lands in the committed filename.
    Runs under both scan modes — the scan path doesn't change the
    parquet shape, but exercising both keeps the gate honest."""
    capture = tmp_path / "capture"
    capture.mkdir()
    _write_capture(capture, "rid", "google/gemma-4-E4B-it")

    result = CliRunner().invoke(
        cli, ["export", str(capture), "--push", "me/d", *scan_args],
    )
    assert result.exit_code == 0, result.output
    op = fake_hf_api.commits[0]["operations"][0]
    assert "gemma-4-E4B-it" in op["path_in_repo"]


def test_export_auto_detect_fails_on_mixed_models(tmp_path: Path):
    """Captures spanning multiple models fail loudly."""
    capture = tmp_path / "capture"
    capture.mkdir()
    _write_capture(capture, "a", "model-1")
    _write_capture(capture, "b", "model-2")

    result = CliRunner().invoke(
        cli, ["export", str(capture), "--push", "me/d"],
    )
    assert result.exit_code != 0
    assert "multiple models" in result.output


def test_export_no_model_in_captures_fails(tmp_path: Path):
    """A capture dir with no model field at all is a hard error."""
    import json
    capture = tmp_path / "capture"
    capture.mkdir()
    (capture / "rid.request.json").write_text(json.dumps({
        "request_id": "rid", "captured_at": 1,
        "body": {"messages": []},
    }))

    result = CliRunner().invoke(
        cli, ["export", str(capture), "--push", "me/d"],
    )
    assert result.exit_code != 0
    assert "no captured requests with a model field" in result.output


def test_export_push_rejects_malformed_dataset_uri(tmp_path: Path):
    capture = tmp_path / "capture"
    capture.mkdir()
    _write_capture(capture, "rid", "m")

    result = CliRunner().invoke(
        cli, ["export", str(capture), "--push", "just-an-owner"],
    )
    assert result.exit_code != 0
    assert "<owner>/<base>" in result.output


def test_export_resolves_workdir_layout_and_reads_agent_from_run_json(
    tmp_path: Path, fake_hf_api, scan_args,
):
    """Pointing export at a workdir uses its captures/ subdir AND picks up
    agent from run.json so the parquet filename embeds the agent."""
    import json
    workdir = tmp_path / "ws" / "hermes-local-20260512-162345"
    captures = workdir / "captures"
    captures.mkdir(parents=True)
    _write_capture(captures, "rid", "google/gemma-4-E4B-it")
    (workdir / "run.json").write_text(json.dumps({"agent": "hermes"}))

    result = CliRunner().invoke(
        cli, ["export", str(workdir), "--push", "me/d", *scan_args],
    )
    assert result.exit_code == 0, result.output
    op = fake_hf_api.commits[0]["operations"][0]
    assert "hermes" in op["path_in_repo"]


def test_export_all_walks_workspace_in_one_commit(
    tmp_path: Path, monkeypatch, fake_hf_api, scan_args,
):
    """--all enumerates every run-id in the workspace and pushes them all
    in one git commit."""
    import json
    monkeypatch.setenv("AGENTCAP_WORKSPACE", str(tmp_path))
    ws = tmp_path / ".agentcap"
    for run_id in ("hermes-local-20260512-160000", "goose-local-20260512-170000"):
        d = ws / run_id / "captures"
        d.mkdir(parents=True)
        _write_capture(d, "rid", "m")
        (ws / run_id / "run.json").write_text(json.dumps({
            "agent": run_id.split("-")[0],
        }))

    result = CliRunner().invoke(
        cli, ["export", "--all", "--push", "me/d", *scan_args],
    )
    assert result.exit_code == 0, result.output
    assert len(fake_hf_api.commits) == 1
    assert len(fake_hf_api.commits[0]["operations"]) == 2
