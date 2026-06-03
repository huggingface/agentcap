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


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def test_get_sandbox_explicit_override_podman():
    """Forcing podman returns a PodmanSandbox keyed on the canonical
    per-agent image ref."""
    from agentcap.sandbox.podman import PodmanSandbox
    sb = get_sandbox(agent="goose", prefer="podman")
    assert isinstance(sb, PodmanSandbox)
    assert sb.image == "localhost/agentcap-goose:latest"


def test_get_sandbox_env_var_selects_podman(monkeypatch):
    """``AGENTCAP_SANDBOX=podman`` overrides OS autodetect."""
    from agentcap.sandbox.podman import PodmanSandbox
    monkeypatch.setenv("AGENTCAP_SANDBOX", "podman")
    sb = get_sandbox(agent="goose")
    assert isinstance(sb, PodmanSandbox)


def test_get_sandbox_prefer_wins_over_env(monkeypatch):
    """``prefer=`` is the test-override knob — it wins over the user-
    facing ``AGENTCAP_SANDBOX`` env var."""
    monkeypatch.setenv("AGENTCAP_SANDBOX", "podman")
    sb = get_sandbox(agent="goose", prefer="bwrap")
    assert isinstance(sb, BwrapSandbox)


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
# Bwrap — end-to-end (Linux + bwrap + buildah)
# ---------------------------------------------------------------------------

# The sandbox primitives (mount semantics, network reachability) are
# agent-agnostic; we exercise them on the opencode image because it
# has the smallest install surface. Any provisioned per-agent image
# would work the same.
_SANDBOX_TEST_AGENT = "opencode"


_HAS_BWRAP_AND_BUILDAH = (
    platform.system() == "Linux"
    and shutil.which("bwrap") is not None
    and shutil.which("buildah") is not None
)


@pytest.mark.skipif(
    not _HAS_BWRAP_AND_BUILDAH,
    reason="Linux + bwrap + buildah required",
)
def test_bwrap_runs_inside_image_rootfs(agentcap_buildah_image_for):
    """End-to-end: bwrap mounts the per-agent image rootfs and the
    agent's binary is on PATH inside the sandbox — but NOT on the
    host (in general). The image-build fixture ensures the image
    exists before the test."""
    tag = agentcap_buildah_image_for(_SANDBOX_TEST_AGENT)
    # opencode-init refuses to start without AGENTCAP_MODEL (it bakes
    # the model id into ``~/.config/opencode/opencode.json``). The
    # value is not exercised by this test, only by the entrypoint.
    env = {"AGENTCAP_MODEL": "test/dummy"}
    with BwrapSandbox(image=tag, env=env) as sb:
        # opencode is installed inside the image at /usr/local/bin/opencode.
        r = sb.run(["which", "opencode"], timeout=60)
        assert r.returncode == 0, r.stderr
        assert r.stdout.strip() == "/usr/local/bin/opencode"


@pytest.mark.skipif(
    not _HAS_BWRAP_AND_BUILDAH,
    reason="Linux + bwrap + buildah required",
)
def test_bwrap_blocks_writes_outside_allowlist(agentcap_buildah_image_for, tmp_path):
    """End-to-end: write to a path in ``writable_paths`` succeeds;
    write outside fails. The image rootfs is read-only and the host
    filesystem (apart from explicitly bound paths) is invisible."""
    tag = agentcap_buildah_image_for(_SANDBOX_TEST_AGENT)
    env = {"AGENTCAP_MODEL": "test/dummy"}
    with BwrapSandbox(image=tag, env=env) as sb:
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
def test_bwrap_image_has_no_host_home_visibility(agentcap_buildah_image_for):
    """End-to-end: host ``$HOME`` content is NOT visible inside the
    image-mounted sandbox. The path may coincidentally exist
    (``ubuntu:24.04`` ships a skeleton ``/home/ubuntu/``) — what
    matters is that the host's actual subtrees aren't there.

    Asserts by probing a path that exists on the host (the agentcap
    repo's pyproject.toml) but cannot exist in the image."""
    tag = agentcap_buildah_image_for(_SANDBOX_TEST_AGENT)
    with BwrapSandbox(image=tag) as sb:
        # cwd of pytest is the agentcap repo root, which lives under the
        # host's $HOME. The repo's pyproject.toml definitely exists on
        # the host; it should be invisible inside the sandbox.
        host_marker = str(Path("pyproject.toml").resolve())
        r = sb.run(["test", "-f", host_marker], timeout=30)
        assert r.returncode != 0, (
            f"Host file {host_marker!r} leaked into the sandbox."
        )
