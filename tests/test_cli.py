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
        yield types.SimpleNamespace(
            host="127.0.0.1", port=18001,
            set_context=lambda **_: None,
        )

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


def _seed_workspace_run(root: Path, run_id: str, rids: list[tuple[str, str]]) -> None:
    """Create a fake workspace run with captures for each (rid, prompt)."""
    import json as _json
    cap = root / ".agentcap" / run_id / "captures"
    cap.mkdir(parents=True)
    for i, (rid, prompt) in enumerate(rids):
        body = {"model": "m", "messages": [{"role": "user", "content": prompt}]}
        (cap / f"{rid}.request.json").write_text(_json.dumps({
            "request_id": rid, "captured_at": 1000 + i,
            "upstream_url": "http://x", "body": body,
        }))
        (cap / f"{rid}.response.json").write_text(_json.dumps({
            "request_id": rid, "captured_at_resp": 1001 + i,
            "status_code": 200, "body": {},
        }))


def _seed_workspace_run_with_meta(
    root: Path, run_id: str, *, agent: str = "hermes", model: str = "m",
) -> None:
    """Like _seed_workspace_run but also writes a minimal run.json so
    the run picker discovers it."""
    import json as _json
    _seed_workspace_run(root, run_id, [("aaa", "p1")])
    (root / ".agentcap" / run_id / "run.json").write_text(_json.dumps({
        "agent": agent, "model": model, "upstream": "http://x",
        "turns_per_task": 1,
        "tasks": [{
            "task_id": "task_01", "prompt": "p1", "completed_turns": 1,
            "turns": [{"turn": 1, "returncode": 0, "duration_s": 1.0}],
        }],
    }))


def test_inspect_resolves_rid_from_workspace(tmp_path: Path, monkeypatch):
    import json as _json
    monkeypatch.setenv("AGENTCAP_WORKSPACE", str(tmp_path))
    cap = tmp_path / ".agentcap" / "hermes-local-20260101-000000" / "captures"
    cap.mkdir(parents=True)
    body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    (cap / "rid.request.json").write_text(_json.dumps({
        "request_id": "rid", "captured_at": 1,
        "upstream_url": "http://x", "body": body,
    }))
    (cap / "rid.response.json").write_text(_json.dumps({
        "request_id": "rid", "captured_at_resp": 2,
        "status_code": 200, "body": {},
    }))

    result = CliRunner().invoke(cli, ["inspect", "rid"])
    assert result.exit_code == 0, result.stderr
    assert _json.loads(result.stdout) == body


def test_inspect_unknown_rid_errors(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AGENTCAP_WORKSPACE", str(tmp_path))
    (tmp_path / ".agentcap").mkdir()
    result = CliRunner().invoke(cli, ["inspect", "ghost"])
    assert result.exit_code != 0
    assert "ghost" in result.output


def test_inspect_run_id_falls_back_to_table_without_fzf(
    tmp_path: Path, monkeypatch
):
    """Without fzf on PATH the picker prints a plain table."""
    monkeypatch.setenv("AGENTCAP_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("PATH", "")
    _seed_workspace_run(
        tmp_path, "hermes-local-20260101-000000",
        [("aaa", "first prompt"), ("bbb", "second prompt")],
    )

    result = CliRunner().invoke(cli, ["inspect", "hermes-local-20260101-000000"])
    assert result.exit_code == 0, result.output
    assert "aaa" in result.output and "bbb" in result.output
    assert "first prompt" in result.output


def test_inspect_no_arg_opens_run_picker(tmp_path: Path, monkeypatch):
    """With no arg, inspect now opens the run picker first. Without fzf
    the run table is printed and inspect exits (user must re-invoke
    with an explicit run-id)."""
    monkeypatch.setenv("AGENTCAP_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("PATH", "")
    # Seed a run with run.json so the run picker finds it.
    _seed_workspace_run_with_meta(
        tmp_path, "hermes-local-20260101-000000",
        agent="hermes", model="m",
    )

    result = CliRunner().invoke(cli, ["inspect"])
    assert result.exit_code == 0, result.output
    # The run table includes the run-id.
    assert "hermes-local-20260101-000000" in result.output


def test_inspect_no_arg_empty_workspace_errors(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AGENTCAP_WORKSPACE", str(tmp_path))
    (tmp_path / ".agentcap").mkdir()
    result = CliRunner().invoke(cli, ["inspect"])
    assert result.exit_code != 0
    assert "no runs" in result.output or "no workspace" in result.output


def test_replay_posts_body_to_target(tmp_path: Path, monkeypatch):
    """Replay POSTs the captured body verbatim and surfaces the response."""
    import json as _json

    monkeypatch.setenv("AGENTCAP_WORKSPACE", str(tmp_path))
    cap = tmp_path / ".agentcap" / "hermes-local-20260101-000000" / "captures"
    cap.mkdir(parents=True)
    body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    (cap / "rid.request.json").write_text(_json.dumps({
        "request_id": "rid", "captured_at": 1,
        "upstream_url": "http://x", "body": body,
    }))

    seen: dict = {}

    class _FakeResponse:
        status_code = 200
        content = b'{"choices":[{"message":{"content":"hello world"}}]}'
        def json(self) -> dict:
            return {"choices": [{"message": {"content": "hello world"}}]}

    def _fake_post(url, json=None, timeout=None):  # noqa: A002
        seen["url"] = url
        seen["json"] = json
        seen["timeout"] = timeout
        return _FakeResponse()

    import httpx
    monkeypatch.setattr(httpx, "post", _fake_post)

    result = CliRunner().invoke(
        cli, ["replay", "rid", "--target", "http://server:9"],
    )
    assert result.exit_code == 0, result.stderr
    assert seen["url"] == "http://server:9/v1/chat/completions"
    assert seen["json"] == body
    # Default rendering: just the assistant's content text.
    assert "hello world" in result.stdout

    # --raw surfaces the pretty JSON dump instead.
    result = CliRunner().invoke(
        cli, ["replay", "rid", "--target", "http://server:9", "--raw"],
    )
    assert result.exit_code == 0, result.stderr
    assert '"choices"' in result.stdout
