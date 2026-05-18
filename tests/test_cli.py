"""CLI smoke tests for `agentcap`.

These do not actually start a uvicorn server — they patch out
``agentcap.proxy.serve`` and assert the right kwargs are computed from
the CLI flags. The proxy itself has its own integration test suite.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from agentcap.__main__ import cli, _parse_listen


def test_help_lists_subcommands():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    for sub in ("proxy", "export", "run"):
        assert sub in result.output


def test_version_flag():
    from agentcap import __version__

    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_proxy_requires_upstream_and_trace_dir():
    runner = CliRunner()
    result = runner.invoke(cli, ["proxy"])
    assert result.exit_code != 0
    assert "--upstream" in result.output


def test_parse_listen_ipv4():
    assert _parse_listen("127.0.0.1:8001") == ("127.0.0.1", 8001)


def test_parse_listen_no_colon_rejected():
    with pytest.raises(Exception):
        _parse_listen("127.0.0.1")


def test_parse_listen_bad_port_rejected():
    with pytest.raises(Exception):
        _parse_listen("127.0.0.1:notaport")


def test_parse_listen_out_of_range_rejected():
    with pytest.raises(Exception):
        _parse_listen("127.0.0.1:99999")


def test_run_requires_agent_upstream_and_workdir():
    runner = CliRunner()
    result = runner.invoke(cli, ["run"])
    assert result.exit_code != 0
    # Click reports the first missing required option
    assert "--agent" in result.output


def test_run_synthesized_defaults_from_upstream_and_model(
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
        yield None

    monkeypatch.setattr("agentcap.proxy.serve_in_thread", fake_proxy)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "run",
            "--agent", "hermes",
            "--model", "google/gemma-4-E4B-it",
            "--upstream", "http://up",
            "--workdir", str(tmp_path / "wd"),
            "--tasks", str(tasks),
            "--turns", "2",
            "--followup", "synthesized",
        ],
    )
    assert result.exit_code == 0, result.output


def test_run_invokes_orchestrator_under_proxy(tmp_path: Path, monkeypatch, fake_sandbox):
    """Smoke-test for the `run` command. Patches out the proxy lifecycle
    and the driver factory so no subprocesses or sockets are touched."""
    import contextlib

    from agentcap.drivers import AgentTurn

    tasks = tmp_path / "tasks.txt"
    tasks.write_text("first task\nsecond task\n")
    workdir = tmp_path / "wd"

    # Fake driver returned by get_driver; records calls.
    class _FakeDriver:
        name = "hermes"

        def __init__(self) -> None:
            self.start_calls = 0
            self.resume_calls = 0

        def start(self, prompt, *, env=None, timeout=None):
            self.start_calls += 1
            return AgentTurn(
                session_id="ses_xyz", response_text="r", returncode=0,
                stdout="", stderr="",
            )

        def resume(self, prompt, *, session_id, env=None, timeout=None):
            self.resume_calls += 1
            return AgentTurn(
                session_id=session_id, response_text="r", returncode=0,
                stdout="", stderr="",
            )

    fake_driver = _FakeDriver()
    monkeypatch.setattr(
        "agentcap.drivers.get_driver", lambda name, **kw: fake_driver
    )

    # Bypass require_sandbox_or_die (which would build an image / boot
    # a VM) with the test-only fake_sandbox fixture.
    monkeypatch.setattr(
        "agentcap.sandbox.require_sandbox_or_die",
        lambda **kw: fake_sandbox,
    )

    proxy_started = {"count": 0}

    @contextlib.contextmanager
    def fake_proxy(*args, **kwargs):
        proxy_started["count"] += 1
        yield None

    monkeypatch.setattr("agentcap.proxy.serve_in_thread", fake_proxy)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "run",
            "--agent", "hermes",
            "--model", "google/gemma-4-E4B-it",
            "--upstream", "http://up:8000",
            "--workdir", str(workdir),
            "--tasks", str(tasks),
            "--turns", "2",
        ],
    )
    assert result.exit_code == 0, result.output
    assert proxy_started["count"] == 1
    assert fake_driver.start_calls == 2   # one per task
    assert fake_driver.resume_calls == 2  # one follow-up per task
    # run.json summary written
    summary_path = workdir / "run.json"
    assert summary_path.is_file()
    import json

    summary = json.loads(summary_path.read_text())
    assert summary["agent"] == "hermes"
    assert len(summary["tasks"]) == 2
    assert summary["tasks"][0]["completed_turns"] == 2


def test_export_requires_output_or_push(tmp_path: Path):
    trace = tmp_path / "trace"
    trace.mkdir()
    runner = CliRunner()
    result = runner.invoke(cli, ["export", str(trace), "--model", "m"])
    assert result.exit_code != 0
    assert "--output" in result.output or "--push" in result.output


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
        yield None

    monkeypatch.setattr("agentcap.proxy.serve_in_thread", fake_proxy)
    monkeypatch.setenv("HF_TOKEN", "hf_env_token")

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "run",
            "--agent", "hermes",
            "--model", "Qwen/Qwen3-8B",
            "--upstream", "https://router.huggingface.co",
            "--workdir", str(tmp_path / "wd"),
            "--tasks", str(tasks),
            "--turns", "1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "HF Router token source=HF_TOKEN" in result.output


def test_export_auto_detects_model_from_traces(tmp_path: Path):
    """When --model is omitted, the CLI infers it from the trace dir."""
    import json

    trace = tmp_path / "trace"
    trace.mkdir()
    # One captured request with the model field populated.
    (trace / "rid.request.json").write_text(
        json.dumps({
            "request_id": "rid",
            "captured_at": 1,
            "body": {
                "model": "google/gemma-4-E4B-it",
                "messages": [{"role": "user", "content": "x"}],
            },
        })
    )

    runner = CliRunner()
    with patch("agentcap.export.export_local", return_value=1):
        result = runner.invoke(
            cli, ["export", str(trace), "--output", str(tmp_path / "out.parquet")]
        )
    assert result.exit_code == 0, result.output
    assert "using model 'google/gemma-4-E4B-it' (auto-detected)" in result.output


def test_export_auto_detect_fails_on_mixed_models(tmp_path: Path):
    """If captures span multiple models, --model becomes mandatory."""
    import json

    trace = tmp_path / "trace"
    trace.mkdir()
    for rid, model in [("a", "model-1"), ("b", "model-2")]:
        (trace / f"{rid}.request.json").write_text(
            json.dumps({
                "request_id": rid,
                "captured_at": 1,
                "body": {"model": model, "messages": []},
            })
        )

    runner = CliRunner()
    result = runner.invoke(
        cli, ["export", str(trace), "--output", str(tmp_path / "out.jsonl")]
    )
    assert result.exit_code != 0
    assert "multiple models" in result.output


def test_export_mixed_models_fail_even_with_model_flag(tmp_path: Path):
    """Datasets never mix models — --model cannot bypass the uniqueness
    check when the trace dir itself spans multiple models."""
    import json

    trace = tmp_path / "trace"
    trace.mkdir()
    for rid, model in [("a", "model-1"), ("b", "model-2")]:
        (trace / f"{rid}.request.json").write_text(
            json.dumps({
                "request_id": rid,
                "captured_at": 1,
                "body": {"model": model, "messages": []},
            })
        )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["export", str(trace), "--model", "explicit-model",
         "--output", str(tmp_path / "out.jsonl")],
    )
    assert result.exit_code != 0
    assert "multiple models" in result.output


def test_export_explicit_model_uses_override_when_traces_uniform(tmp_path: Path):
    """If the trace dir is uniform but --model differs, the override is
    accepted (--model is now only a bucket-filename hint; the parquet's
    per-row ``model`` column still reflects the captured body.model)."""
    import json

    trace = tmp_path / "trace"
    trace.mkdir()
    (trace / "rid.request.json").write_text(
        json.dumps({
            "request_id": "rid",
            "captured_at": 1,
            "body": {"model": "trace-model", "messages": []},
        })
    )

    runner = CliRunner()
    with patch("agentcap.export.export_local", return_value=1):
        result = runner.invoke(
            cli,
            ["export", str(trace), "--model", "override-model",
             "--output", str(tmp_path / "out.parquet")],
        )
    assert result.exit_code == 0, result.output
    # No auto-detect log when --model is explicit
    assert "auto-detected" not in result.output


def test_export_no_model_in_traces_requires_model_flag(tmp_path: Path):
    """Trace dir with no model field at all → --model becomes mandatory."""
    import json

    trace = tmp_path / "trace"
    trace.mkdir()
    (trace / "rid.request.json").write_text(
        json.dumps({
            "request_id": "rid",
            "captured_at": 1,
            "body": {"messages": []},  # no model field
        })
    )

    runner = CliRunner()
    result = runner.invoke(
        cli, ["export", str(trace), "--output", str(tmp_path / "out.jsonl")]
    )
    assert result.exit_code != 0
    assert "no captured requests with a model field" in result.output


def test_export_push_rejects_dataset_repo_uri(tmp_path: Path):
    """Only bucket URIs are accepted by --push; dataset repo URIs must
    fail with a message pointing at the local-export workflow."""
    import json

    trace = tmp_path / "trace"
    trace.mkdir()
    (trace / "rid.request.json").write_text(
        json.dumps({
            "request_id": "rid",
            "captured_at": 1,
            "body": {"model": "m", "messages": []},
        })
    )

    runner = CliRunner()
    result = runner.invoke(
        cli, ["export", str(trace), "--push", "hf://org/some-dataset"]
    )
    assert result.exit_code != 0
    assert "bucket URIs" in result.output
    assert "hf upload" in result.output


def test_export_rejects_both_output_and_push(tmp_path: Path):
    """The two destinations are mutually exclusive."""
    import json

    trace = tmp_path / "trace"
    trace.mkdir()
    (trace / "rid.request.json").write_text(
        json.dumps({
            "request_id": "rid",
            "captured_at": 1,
            "body": {"model": "m", "messages": []},
        })
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["export", str(trace),
         "--output", str(tmp_path / "out.parquet"),
         "--push", "hf://buckets/me/b/x"],
    )
    assert result.exit_code != 0
    assert "not both" in result.output
