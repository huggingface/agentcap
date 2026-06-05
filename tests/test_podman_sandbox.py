"""Structural tests for :mod:`agentcap.sandbox.podman`.

Argv-assembly only — these don't shell out to podman. End-to-end
coverage against a real ``podman run`` lives in
``tests/test_drivers_live.py`` via the live driver tests.
"""

from __future__ import annotations

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


def test_run_names_container_and_force_removes_it(monkeypatch):
    """Every ``run()`` must inject ``--name agentcap-<hex>`` and, in a
    ``finally`` block, fire ``podman rm -f <same-name>`` even when the
    main subprocess succeeded — ``--rm`` only fires on a clean container
    exit, so this is the guarantee against orphaned containers when
    timeouts/kills/dead parents prevent that."""
    import subprocess
    from agentcap.sandbox import podman as pmod

    calls: list[list[str]] = []

    class _Completed:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(argv, **_kw):
        calls.append(list(argv))
        return _Completed()

    monkeypatch.setattr(subprocess, "run", fake_run)
    sb = pmod.PodmanSandbox(image="img:latest")
    sb.run(["echo", "hi"])

    assert len(calls) == 2, f"expected run + rm; got {calls!r}"
    run_cmd, rm_cmd = calls
    # ``--name <agentcap-...>`` was inserted right after ``podman run``.
    assert "--name" in run_cmd
    name_idx = run_cmd.index("--name")
    name = run_cmd[name_idx + 1]
    assert name.startswith("agentcap-")
    # The cleanup targets the same name.
    assert rm_cmd[:3] == ["podman", "rm", "-f"]
    assert rm_cmd[3] == name


def test_run_force_removes_container_even_if_subprocess_raises(monkeypatch):
    """When ``subprocess.run`` raises (e.g. ``TimeoutExpired``), the
    container can still be alive — the cleanup ``podman rm -f`` must
    fire from the ``finally`` so the orchestrator never leaks a
    container even on timeout / SIGINT. The cleanup must also target
    the SAME name that ``podman run`` was given; removing the wrong
    container would silently nuke something else."""
    import subprocess
    from agentcap.sandbox import podman as pmod

    run_calls: list[list[str]] = []
    rm_calls: list[list[str]] = []

    def fake_run(argv, **kw):
        if argv[:3] == ["podman", "rm", "-f"]:
            rm_calls.append(list(argv))
            class _R:
                returncode = 0
                stdout = ""
                stderr = ""
            return _R()
        run_calls.append(list(argv))
        raise subprocess.TimeoutExpired(cmd=argv, timeout=kw.get("timeout"))

    monkeypatch.setattr(subprocess, "run", fake_run)
    sb = pmod.PodmanSandbox(image="img:latest")
    try:
        sb.run(["sleep", "60"], timeout=0.01)
    except subprocess.TimeoutExpired:
        pass
    else:
        raise AssertionError("expected TimeoutExpired to propagate")

    # Extract the name passed to ``podman run`` via ``--name <X>``.
    assert len(run_calls) == 1, run_calls
    run_argv = run_calls[0]
    assert "--name" in run_argv
    run_name = run_argv[run_argv.index("--name") + 1]
    assert run_name.startswith("agentcap-")

    # Cleanup must have targeted that exact name — not some other
    # container, and not no container.
    assert len(rm_calls) == 1, rm_calls
    assert rm_calls[0] == ["podman", "rm", "-f", run_name]


def test_run_propagates_main_failure_when_cleanup_also_fails(monkeypatch):
    """If both the main ``podman run`` and the cleanup ``podman rm -f``
    raise, callers must see the ORIGINAL exception — not the cleanup's.
    Otherwise a transient ``rm`` failure would mask the real reason the
    container run failed (timeout, exit code, etc.)."""
    import subprocess
    from agentcap.sandbox import podman as pmod

    def fake_run(argv, **kw):
        if argv[:3] == ["podman", "rm", "-f"]:
            raise RuntimeError("cleanup boom")
        raise subprocess.TimeoutExpired(cmd=argv, timeout=kw.get("timeout"))

    monkeypatch.setattr(subprocess, "run", fake_run)
    sb = pmod.PodmanSandbox(image="img:latest")
    try:
        sb.run(["sleep", "60"], timeout=0.01)
    except subprocess.TimeoutExpired:
        pass  # original exception preserved
    except RuntimeError as exc:
        raise AssertionError(
            f"cleanup exception ({exc}) leaked past the finally — "
            f"primary TimeoutExpired was masked"
        )
    else:
        raise AssertionError("expected TimeoutExpired to propagate")
