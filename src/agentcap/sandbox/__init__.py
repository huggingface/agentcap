"""Filesystem / network sandbox for capture-run subprocesses.

Backends:

  * podman (default on Linux and macOS) — :class:`PodmanSandbox`,
    one container per ``run()``, built from
    ``containers/agentcap-<agent>.Containerfile``.
  * bwrap (Linux only, opt-in via ``AGENTCAP_SANDBOX=bwrap``) —
    :class:`BwrapSandbox`, namespace-only sandbox on the host kernel.

``agentcap run`` requires a real sandbox on every supported host —
the agent CLI lives inside the per-agent image, never on the host.
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
    :meth:`write_text` / :meth:`read_text` are host paths bind-mounted
    into the agent's view at the same path."""

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


def get_sandbox(
    *,
    agent: str,
    prefer: str | None = None,
    env: dict[str, str] | None = None,
    readonly_paths: list[Path] | None = None,
    writable_paths: list[Path] | None = None,
) -> Sandbox:
    """Pick the sandbox backend for the current host. Pure: does not
    build images.

    ``prefer`` forces a specific backend (``"bwrap"`` or ``"podman"``);
    useful for tests on a host where autodetect would pick the other.
    Callers must call :func:`require_sandbox_or_die` to provision the
    runtime before using the returned sandbox.
    """
    from .bwrap import BwrapSandbox
    from .image_provisioning import image_tag
    from .podman import PodmanSandbox

    backend = prefer or _autodetect_backend()

    if backend == "bwrap":
        return BwrapSandbox(
            image=image_tag(agent), env=env,
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
        f"expected 'bwrap' or 'podman'"
    )


def _autodetect_backend() -> str:
    """``AGENTCAP_SANDBOX`` env var wins over the OS default so users
    can switch backends without code changes. ``prefer=`` on
    :func:`get_sandbox` wins over both — it's the test-override knob."""
    env_choice = os.environ.get("AGENTCAP_SANDBOX")
    if env_choice:
        return env_choice
    system = platform.system()
    if system in ("Linux", "Darwin"):
        return "podman"
    raise NotImplementedError(
        f"agentcap sandboxing is only supported on Linux and macOS; "
        f"host is {system!r}."
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
    """Return a real sandbox for the resolved backend, or exit 2 with
    an install hint. Triggers an image build on first use."""
    import sys

    backend = _autodetect_backend()

    if backend == "bwrap":
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
            agent=agent, prefer=backend, env=env,
            readonly_paths=readonly_paths,
            writable_paths=writable_paths,
        )

    if backend == "podman":
        if not shutil.which("podman"):
            sys.stderr.write(
                f"{command}: podman is required for the podman backend.\n"
                "    Install with: brew install podman (macOS) "
                "or apt install podman (Linux)\n"
            )
            sys.exit(2)
        from .podman_provisioning import ensure_image, ensure_machine_running
        try:
            ensure_machine_running(log=log)
        except RuntimeError as exc:
            sys.stderr.write(f"{command}: {exc}\n")
            sys.exit(2)
        ensure_image(agent, log=log)
        return get_sandbox(
            agent=agent, prefer=backend, env=env,
            readonly_paths=readonly_paths,
            writable_paths=writable_paths,
        )

    raise NotImplementedError(
        f"unknown sandbox backend {backend!r}"
    )


__all__ = [
    "Sandbox",
    "get_sandbox",
    "require_sandbox_or_die",
]
