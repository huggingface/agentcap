"""Tests for the sandbox abstraction."""

from __future__ import annotations

import pytest

from agentcap.sandbox import get_sandbox
from agentcap.sandbox.podman import PodmanSandbox


def test_get_sandbox_returns_podman_sandbox():
    """The factory hands back a ``PodmanSandbox`` keyed on the
    canonical per-agent image ref."""
    sb = get_sandbox(agent="goose")
    assert isinstance(sb, PodmanSandbox)
    assert sb.image == "localhost/agentcap-goose:latest"


def test_get_sandbox_requires_agent():
    with pytest.raises(TypeError):
        get_sandbox()  # type: ignore[call-arg]
