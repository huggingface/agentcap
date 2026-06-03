"""Filesystem / network sandbox for capture-run subprocesses.

Backends:

  * Linux                          -> :class:`BwrapSandbox`, mounted on
                                       a per-agent buildah image (see
                                       :func:`image_provisioning.ensure_image`).
  * macOS, ``limactl`` on PATH     -> :class:`LimaSandbox`, per-agent VM.

There is no fallback. ``agentcap run`` requires a real sandbox on
every supported host — the agent CLI lives inside the per-agent
image/VM, never on the host.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class Sandbox(Protocol):
    """Paths returned by :meth:`mkdtemp` and consumed by
    :meth:`write_text` / :meth:`read_text` are sandbox-side: host
    paths on Linux/bwrap (bind-mounted into the agent's view at the
    same path), VM-side paths on macOS/Lima."""

    name: str

    def wrap(
        self,
        argv: list[str],
        *,
        writable_paths: list[Path],
        deny_network: bool = False,
    ) -> list[str]:
        ...

    def run(
        self,
        argv: list[str],
        *,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        writable_paths: list[Path] | None = None,
        deny_network: bool = False,
        timeout: float | None = None,
        check: bool = False,
    ) -> subprocess.CompletedProcess:
        ...

    def mkdtemp(self, prefix: str = "agentcap-") -> str: ...
    def rmtree(self, path: str) -> None: ...
    def write_text(self, path: str, content: str) -> None: ...
    def read_text(self, path: str) -> str: ...


def lima_vm_name(agent: str) -> str:
    """Canonical Lima VM name for a given agent."""
    return f"agentcap-{agent}"


def get_sandbox(
    *,
    agent: str,
    prefer: str | None = None,
    env: dict[str, str] | None = None,
    readonly_paths: list[Path] | None = None,
    writable_paths: list[Path] | None = None,
) -> Sandbox:
    """Pick the sandbox backend for the current host. Pure: does not
    build images or boot VMs.

    ``agent`` is required — every backend is per-agent (the image
    tag / VM name encodes it). ``prefer`` forces a specific backend
    (``"bwrap"`` or ``"lima"``); useful for tests on a host where
    autodetect would pick the other.

    ``env`` is overlaid on top of the per-image baked env vars
    (both backends). Used by ``agentcap run`` to pass things like
    ``AGENTCAP_PROXY_URL`` that the image's startup script reads.

    Callers must call :func:`require_sandbox_or_die` (or
    :func:`image_provisioning.ensure_image` /
    :func:`lima_provisioning.ensure_vm` directly) to actually
    provision the runtime before using the returned sandbox.
    """
    from .bwrap import BwrapSandbox
    from .image_provisioning import image_tag
    from .lima import LimaSandbox
    from .podman import PodmanSandbox

    backend = prefer or _autodetect_backend()

    if backend == "bwrap":
        return BwrapSandbox(
            image=image_tag(agent), env=env,
            readonly_paths=readonly_paths,
            writable_paths=writable_paths,
        )

    if backend == "lima":
        return LimaSandbox(
            vm=lima_vm_name(agent), env=env,
            readonly_paths=readonly_paths,
            writable_paths=writable_paths,
        )

    if backend == "podman":
        return PodmanSandbox(
            image=f"localhost/agentcap-{agent}:latest", env=env,
            readonly_paths=readonly_paths,
            writable_paths=writable_paths,
        )

    raise ValueError(
        f"unknown sandbox backend {backend!r}; "
        f"expected 'bwrap', 'lima', or 'podman'"
    )


def _autodetect_backend() -> str:
    """``AGENTCAP_SANDBOX`` env var wins over OS detection so users can
    switch backends without code changes. ``prefer=`` on
    :func:`get_sandbox` wins over both — it's the test-override knob."""
    env_choice = os.environ.get("AGENTCAP_SANDBOX")
    if env_choice:
        return env_choice
    system = platform.system()
    if system == "Linux":
        return "bwrap"
    if system == "Darwin":
        return "lima"
    raise NotImplementedError(
        f"agentcap sandboxing is only supported on Linux (bwrap) and "
        f"macOS (lima); host is {system!r}."
    )


def require_sandbox_or_die(
    *,
    agent: str,
    command: str = "agentcap run",
    log=lambda msg: None,
    env: dict[str, str] | None = None,
    readonly_paths: list[Path] | None = None,
    writable_paths: list[Path] | None = None,
) -> "Sandbox":
    """Return a real sandbox for the current host, or exit 2 with an
    install hint. Triggers an image / VM build on first use.

    On Linux a missing ``bwrap`` or ``buildah`` causes exit 2 with a
    one-line ``apt install`` hint. On macOS a missing ``limactl``
    likewise.
    """
    import sys

    system = platform.system()
    if system == "Linux":
        missing = [
            tool for tool in ("bwrap", "buildah") if not shutil.which(tool)
        ]
        if missing:
            sys.stderr.write(
                f"{command}: missing required tools on Linux: "
                f"{', '.join(missing)}\n"
                f"    Install with: apt install {' '.join(missing)}\n"
                "    bubblewrap also needs unprivileged user namespaces — "
                "see README 'Sandbox prerequisites' for Ubuntu 24.04.\n"
            )
            sys.exit(2)
        from .image_provisioning import ensure_image
        ensure_image(agent, log=log)
        return get_sandbox(
            agent=agent, env=env,
            readonly_paths=readonly_paths,
            writable_paths=writable_paths,
        )
    if system == "Darwin":
        if not shutil.which("limactl"):
            sys.stderr.write(
                f"{command}: Lima is required on macOS for agent sandboxing.\n"
                "    Install with: brew install lima\n"
            )
            sys.exit(2)
        from .lima_provisioning import ensure_vm
        ensure_vm(
            agent, log=log,
            readonly_paths=readonly_paths,
            writable_paths=writable_paths,
        )
        return get_sandbox(
            agent=agent, env=env,
            readonly_paths=readonly_paths,
            writable_paths=writable_paths,
        )
    raise NotImplementedError(
        f"agentcap sandboxing is only supported on Linux (bwrap) and "
        f"macOS (lima); host is {system!r}."
    )


__all__ = [
    "Sandbox",
    "get_sandbox",
    "lima_vm_name",
    "require_sandbox_or_die",
]
