"""Tests for the sandbox abstraction.

Structural tests run anywhere. End-to-end enforcement tests are
gated on the relevant host primitive being available — the bwrap
tests require Linux + ``bwrap`` + ``buildah`` and skip otherwise.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from agentcap.sandbox import (
    Sandbox,
    get_sandbox,
)
from agentcap.sandbox.bwrap import BwrapSandbox, build_command
from agentcap.sandbox.lima import LimaSandbox, build_command as build_lima_command


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def test_get_sandbox_explicit_override_lima():
    """Forcing lima skips the host autodetect; the result is a
    LimaSandbox with the canonical per-agent VM name."""
    sb = get_sandbox(agent="goose", prefer="lima")
    assert isinstance(sb, LimaSandbox)
    assert sb.vm == "agentcap-goose"


def test_get_sandbox_rejects_unknown():
    with pytest.raises(ValueError, match="unknown sandbox backend"):
        get_sandbox(agent="goose", prefer="firejail")


def test_get_sandbox_rejects_sandbox_exec():
    """sandbox-exec was intentionally dropped; asking for it by name
    fails loud so callers can't accidentally rely on a deprecated
    primitive."""
    with pytest.raises(ValueError, match="unknown sandbox backend"):
        get_sandbox(agent="goose", prefer="sandbox-exec")


def test_get_sandbox_requires_agent():
    """``get_sandbox`` has no host-independent path — every backend
    is per-agent."""
    with pytest.raises(TypeError):
        get_sandbox()  # type: ignore[call-arg]


def test_lima_sandbox_implements_protocol():
    assert isinstance(LimaSandbox(), Sandbox)


def test_bwrap_sandbox_implements_protocol():
    assert isinstance(BwrapSandbox(image="agentcap-goose:latest"), Sandbox)


# ---------------------------------------------------------------------------
# Bwrap — argv assembly
# ---------------------------------------------------------------------------

_TEST_CTR = "ctr-abc123"


def test_bwrap_command_wraps_in_buildah_unshare(tmp_path):
    """``build_command`` emits a ``buildah unshare bash -c <script>``
    invocation. The script mounts the existing container, bwraps
    into it, and umounts on exit — without removing the container so
    it survives across multiple runs (BwrapSandbox.close() owns rm).
    """
    cmd = build_command(
        ["echo", "hi"],
        container_id=_TEST_CTR,
        writable_paths=[tmp_path],
        deny_network=True,
    )
    assert cmd[:3] == ["buildah", "unshare", "bash"]
    assert cmd[3] == "-c"
    script = cmd[4]
    assert f"buildah mount {_TEST_CTR}" in script
    assert f"buildah umount {_TEST_CTR}" in script
    # NO `buildah rm` and NO `buildah from` — the container's
    # lifecycle is owned by BwrapSandbox, not by each run().
    assert "buildah rm" not in script
    assert "buildah from" not in script
    # bwrap stays as a child of the shell (not exec'd) so the EXIT
    # trap fires and the namespace-local mount gets torn down.
    assert "exec bwrap" not in script
    assert "bwrap --bind" in script
    assert '--bind "$mnt" /' in script
    assert "--unshare-net" in script
    # original argv survives
    assert "echo hi" in script
    # writable path is bind-mounted
    assert f"--bind {tmp_path.resolve()} {tmp_path.resolve()}" in script


def test_bwrap_command_share_net_when_allowed(tmp_path):
    cmd = build_command(
        ["true"],
        container_id=_TEST_CTR,
        writable_paths=[tmp_path],
        deny_network=False,
    )
    script = cmd[4]
    assert "--share-net" in script
    assert "--unshare-net" not in script


def test_bwrap_command_propagates_env_via_setenv(tmp_path):
    """``env`` kwarg becomes ``--setenv KEY VAL`` flags inside bwrap,
    NOT host-side env vars on the buildah unshare process."""
    cmd = build_command(
        ["true"],
        container_id=_TEST_CTR,
        writable_paths=[tmp_path],
        env={"OPENAI_API_KEY": "dummy", "HF_TOKEN": "secret"},
    )
    script = cmd[4]
    assert "--setenv OPENAI_API_KEY dummy" in script
    assert "--setenv HF_TOKEN secret" in script


def test_bwrap_command_propagates_cwd(tmp_path):
    cmd = build_command(
        ["true"],
        container_id=_TEST_CTR,
        writable_paths=[tmp_path],
        cwd="/work",
    )
    script = cmd[4]
    assert "--chdir /work" in script


# ---------------------------------------------------------------------------
# Lima — argv assembly
# ---------------------------------------------------------------------------


def test_lima_wraps_argv_in_limactl_shell():
    """LimaSandbox.wrap() emits a flat `limactl shell <vm> -- argv`
    — no inner bwrap. The VM is the sandbox."""
    cmd = build_lima_command(["echo", "hi"], vm="agentcap-opencode")
    assert cmd[0] == "limactl"
    assert cmd[1] == "shell"
    assert "agentcap-opencode" in cmd
    sep = cmd.index("--")
    assert cmd[sep + 1 :] == ["echo", "hi"]
    # No bwrap layer inside.
    assert "bwrap" not in cmd


def test_lima_passes_workdir_to_limactl(tmp_path):
    cmd = build_lima_command(["true"], workdir=tmp_path)
    assert "--workdir" in cmd
    i = cmd.index("--workdir")
    assert cmd[i + 1] == str(tmp_path)


def test_lima_respects_custom_vm_name():
    cmd = build_lima_command(["true"], vm="my-vm")
    assert "my-vm" in cmd
    assert "agentcap-unset" not in cmd


def test_lima_deny_network_warns_and_no_ops(tmp_path, capfd):
    """Lima has no per-call network policy; deny_network=True must
    warn to stderr (so callers can see it) and proceed without
    changing the argv."""
    sb = LimaSandbox()
    argv_with = sb.wrap(["true"], writable_paths=[tmp_path], deny_network=True)
    argv_without = sb.wrap(["true"], writable_paths=[tmp_path], deny_network=False)
    assert argv_with == argv_without
    err = capfd.readouterr().err
    assert "deny_network" in err
    assert "Lima" in err


# ---------------------------------------------------------------------------
# Lima — end-to-end (uses lima_vm_for fixture in conftest)
# ---------------------------------------------------------------------------

# The sandbox primitives (mount semantics, network reachability) are
# agent-agnostic; we exercise them on the opencode VM because it has
# the smallest install surface (`curl … | bash`). Any provisioned
# per-agent VM would work the same.
_SANDBOX_TEST_AGENT = "opencode"


def test_lima_vm_can_reach_host_server(lima_vm_for, mock_http_server):
    """End-to-end: a process inside the Lima VM can reach a server
    running on the Mac host via ``host.lima.internal``. This is the
    same network path the capture proxy will use when the agent
    runs inside the VM — without it, the captured-trace flow is
    structurally impossible. Asserts both that the VM-side HTTP call
    succeeds AND that the host-side process saw the request."""
    vm = lima_vm_for(_SANDBOX_TEST_AGENT)
    port, received = mock_http_server
    sb = LimaSandbox(vm=vm)
    code = (
        "import urllib.request, sys; "
        f"r = urllib.request.urlopen("
        f"'http://host.lima.internal:{port}/ping', timeout=10); "
        "sys.stdout.write(r.read().decode())"
    )
    r = subprocess.run(
        sb.wrap(["python3", "-c", code], writable_paths=[]),
        capture_output=True, text=True, timeout=30,
    )
    assert r.returncode == 0, r.stderr
    assert '"ok": true' in r.stdout
    assert "/ping" in received, received


def test_lima_fs_methods_inside_vm(lima_vm_for):
    """End-to-end: ``mkdtemp`` / ``write_text`` / ``read_text`` /
    ``rmtree`` on LimaSandbox execute inside the VM via
    ``limactl shell -- mktemp / cat / base64 / rm``."""
    vm = lima_vm_for(_SANDBOX_TEST_AGENT)
    sb = LimaSandbox(vm=vm)

    d = sb.mkdtemp(prefix="agentcap-test-")
    try:
        assert d.startswith("/tmp/agentcap-test-")
        assert not Path(d).exists()
        payload = 'first\n"quoted"\n$ENV_VAR\n'
        sb.write_text(f"{d}/file.txt", payload)
        assert sb.read_text(f"{d}/file.txt") == payload
    finally:
        sb.rmtree(d)
    r = sb.run(["test", "-d", d])
    assert r.returncode != 0


def test_lima_run_propagates_env_into_vm(lima_vm_for):
    vm = lima_vm_for(_SANDBOX_TEST_AGENT)
    sb = LimaSandbox(vm=vm)
    r = sb.run(["sh", "-c", "echo $HF_TOKEN"], env={"HF_TOKEN": "tok123"})
    assert r.returncode == 0
    assert r.stdout.strip() == "tok123"


def test_lima_run_uses_cwd_inside_vm(lima_vm_for):
    vm = lima_vm_for(_SANDBOX_TEST_AGENT)
    sb = LimaSandbox(vm=vm)
    d = sb.mkdtemp(prefix="lima-cwd-test-")
    try:
        r = sb.run(["pwd"], cwd=d)
        assert r.returncode == 0
        assert r.stdout.strip() == d
    finally:
        sb.rmtree(d)


def test_lima_vm_has_no_host_filesystem_visibility(lima_vm_for):
    """End-to-end: the VM is fully isolated from the Mac filesystem.
    Per-agent templates declare ``mounts: []``, so no Mac path is
    visible inside the VM — neither the user's $HOME nor pytest's
    tmp dir."""
    vm = lima_vm_for(_SANDBOX_TEST_AGENT)
    sb = LimaSandbox(vm=vm)
    home = str(Path.home())
    r = subprocess.run(
        sb.wrap(["mount", "-t", "virtiofs"], writable_paths=[]),
        capture_output=True, text=True, timeout=10,
    )
    if home in r.stdout:
        pytest.skip(
            f"VM {vm!r} was created with a `~` mount (pre-mounts:[] "
            f"template). Delete it (`limactl delete {vm}`) and re-run."
        )
    r = subprocess.run(
        sb.wrap(["ls", home], writable_paths=[]),
        capture_output=True, text=True, timeout=10,
    )
    assert r.returncode != 0


# ---------------------------------------------------------------------------
# Bwrap — end-to-end (Linux + bwrap + buildah)
# ---------------------------------------------------------------------------


_HAS_BWRAP_AND_BUILDAH = (
    platform.system() == "Linux"
    and shutil.which("bwrap") is not None
    and shutil.which("buildah") is not None
)


@pytest.mark.skipif(
    not _HAS_BWRAP_AND_BUILDAH,
    reason="Linux + bwrap + buildah required",
)
def test_bwrap_runs_inside_image_rootfs(agentcap_image_for):
    """End-to-end: bwrap mounts the per-agent image rootfs and the
    agent's binary is on PATH inside the sandbox — but NOT on the
    host (in general). The image-build fixture ensures the image
    exists before the test."""
    tag = agentcap_image_for(_SANDBOX_TEST_AGENT)
    with BwrapSandbox(image=tag) as sb:
        # opencode is installed inside the image at /usr/local/bin/opencode.
        r = sb.run(["which", "opencode"], timeout=60)
        assert r.returncode == 0, r.stderr
        assert r.stdout.strip() == "/usr/local/bin/opencode"


@pytest.mark.skipif(
    not _HAS_BWRAP_AND_BUILDAH,
    reason="Linux + bwrap + buildah required",
)
def test_bwrap_blocks_writes_outside_allowlist(agentcap_image_for, tmp_path):
    """End-to-end: write to a path in ``writable_paths`` succeeds;
    write outside fails. The image rootfs is read-only and the host
    filesystem (apart from explicitly bound paths) is invisible."""
    tag = agentcap_image_for(_SANDBOX_TEST_AGENT)
    with BwrapSandbox(image=tag) as sb:
        inside = tmp_path / "inside.txt"
        r = sb.run(
            ["python3", "-c", f"open({str(inside)!r}, 'w').write('ok')"],
            writable_paths=[tmp_path],
            timeout=30,
        )
        assert r.returncode == 0, r.stderr
        assert inside.read_text() == "ok"

        outside_root = Path(tempfile.mkdtemp(prefix="sb-outside-"))
        try:
            outside = outside_root / "leaked.txt"
            r = sb.run(
                ["python3", "-c", f"open({str(outside)!r}, 'w').write('leak')"],
                writable_paths=[tmp_path],
                timeout=30,
            )
            # outside_root isn't in writable_paths → not bound → not visible.
            assert r.returncode != 0
            assert not outside.exists()
        finally:
            shutil.rmtree(outside_root, ignore_errors=True)


@pytest.mark.skipif(
    not _HAS_BWRAP_AND_BUILDAH,
    reason="Linux + bwrap + buildah required",
)
def test_bwrap_image_has_no_host_home_visibility(agentcap_image_for):
    """End-to-end: host ``$HOME`` content is NOT visible inside the
    image-mounted sandbox. The path may coincidentally exist
    (``ubuntu:24.04`` ships a skeleton ``/home/ubuntu/``) — what
    matters is that the host's actual subtrees aren't there.

    Asserts by probing a path that exists on the host (the agentcap
    repo's pyproject.toml) but cannot exist in the image."""
    tag = agentcap_image_for(_SANDBOX_TEST_AGENT)
    with BwrapSandbox(image=tag) as sb:
        # cwd of pytest is the agentcap repo root, which lives under the
        # host's $HOME. The repo's pyproject.toml definitely exists on
        # the host; it should be invisible inside the sandbox.
        host_marker = str(Path("pyproject.toml").resolve())
        r = sb.run(["test", "-f", host_marker], timeout=30)
        assert r.returncode != 0, (
            f"Host file {host_marker!r} leaked into the sandbox."
        )
