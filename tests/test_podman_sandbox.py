"""Structural tests for :mod:`agentcap.sandbox.podman`.

Argv-assembly tests run anywhere — they don't shell out to podman.
End-to-end execution against a real ``podman run`` lands once
podman is wired into the test fixtures (commit 4).
"""

from __future__ import annotations

import pytest

from agentcap.sandbox import Sandbox
from agentcap.sandbox.podman import PodmanSandbox, build_command


def test_podman_sandbox_implements_protocol():
    assert isinstance(PodmanSandbox(image="agentcap-goose:latest"), Sandbox)


def test_build_command_minimal():
    cmd = build_command(
        ["echo", "hi"],
        image="img:latest",
        writable_paths=[],
    )
    assert cmd[:3] == ["podman", "run", "--rm"]
    assert cmd[-3:] == ["img:latest", "echo", "hi"]


def test_build_command_writable_bind_mount(tmp_path):
    cmd = build_command(
        ["true"],
        image="img:latest",
        writable_paths=[tmp_path],
    )
    expected = f"type=bind,src={tmp_path.resolve()},dst={tmp_path.resolve()}"
    assert "--mount" in cmd
    assert expected in cmd


def test_build_command_readonly_bind_mount(tmp_path):
    cmd = build_command(
        ["true"],
        image="img:latest",
        writable_paths=[],
        readonly_paths=[tmp_path],
    )
    expected = (
        f"type=bind,src={tmp_path.resolve()},dst={tmp_path.resolve()},ro"
    )
    assert expected in cmd


def test_build_command_deny_network():
    cmd = build_command(
        ["true"], image="img:latest", writable_paths=[],
        deny_network=True,
    )
    assert "--network=none" in cmd


def test_build_command_propagates_env():
    cmd = build_command(
        ["true"], image="img:latest", writable_paths=[],
        env={"FOO": "bar"},
    )
    assert "-e" in cmd
    assert "FOO=bar" in cmd


def test_build_command_propagates_cwd(tmp_path):
    cmd = build_command(
        ["true"], image="img:latest", writable_paths=[],
        cwd=str(tmp_path),
    )
    assert "--workdir" in cmd
    assert str(tmp_path) in cmd
    # ``cwd`` is also added to the writable bind set so chdir
    # resolves inside the container.
    expected = f"type=bind,src={tmp_path.resolve()},dst={tmp_path.resolve()}"
    assert expected in cmd


def test_build_command_dedups_overlapping_mounts(tmp_path):
    cmd = build_command(
        ["true"], image="img:latest",
        writable_paths=[tmp_path, tmp_path],
        readonly_paths=[tmp_path],
    )
    mount_args = [a for a in cmd if a.startswith("type=bind,")]
    assert len(mount_args) == 1


def test_wrap_layers_constructor_env_under_call_env(tmp_path):
    sb = PodmanSandbox(image="img:latest", env={"A": "1", "B": "2"})
    cmd = sb.wrap(["true"], writable_paths=[], env={"B": "override"})
    assert "A=1" in cmd
    assert "B=override" in cmd
    assert "B=2" not in cmd


def test_wrap_combines_lifetime_and_per_call_writable_paths(tmp_path):
    lifetime = tmp_path / "lifetime"
    lifetime.mkdir()
    per_call = tmp_path / "percall"
    per_call.mkdir()
    sb = PodmanSandbox(image="img:latest", writable_paths=[lifetime])
    cmd = sb.wrap(["true"], writable_paths=[per_call])
    assert f"type=bind,src={lifetime.resolve()},dst={lifetime.resolve()}" in cmd
    assert f"type=bind,src={per_call.resolve()},dst={per_call.resolve()}" in cmd


def test_close_is_noop():
    sb = PodmanSandbox(image="img:latest")
    sb.close()
    sb.close()


def test_context_manager_closes():
    with PodmanSandbox(image="img:latest") as sb:
        assert sb.image == "img:latest"
