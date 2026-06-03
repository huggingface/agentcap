"""Tests for the sandbox abstraction."""

from __future__ import annotations

import pytest

from agentcap.sandbox import get_sandbox


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
    """``AGENTCAP_SANDBOX=podman`` resolves through the factory."""
    from agentcap.sandbox.podman import PodmanSandbox
    monkeypatch.setenv("AGENTCAP_SANDBOX", "podman")
    sb = get_sandbox(agent="goose")
    assert isinstance(sb, PodmanSandbox)


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
