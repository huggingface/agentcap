"""Filesystem / network sandbox for capture-run subprocesses.

Single implementation: each ``run()`` is an ephemeral
``podman run --rm`` against the per-agent image built from
``containers/agentcap-<agent>.Containerfile``. The agent CLI lives
inside the image, never on the host.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
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
    env: dict[str, str] | None = None,
    readonly_paths: list[Path] | None = None,
    writable_paths: list[Path] | None = None,
) -> Sandbox:
    """Return a sandbox handle for ``agent``. Pure: does not build
    the image. Call :func:`require_sandbox_or_die` to provision."""
    from .podman import PodmanSandbox
    from .podman_provisioning import image_tag
    return PodmanSandbox(
        image=image_tag(agent), env=env,
        readonly_paths=readonly_paths,
        writable_paths=writable_paths,
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
    """Return a sandbox handle, or exit 2 with an install hint.
    Triggers an image build on first use."""
    system = platform.system()
    if system not in ("Linux", "Darwin"):
        sys.stderr.write(
            f"{command}: agentcap sandboxing is only supported on "
            f"Linux and macOS; host is {system!r}.\n"
        )
        sys.exit(2)
    if not shutil.which("podman"):
        sys.stderr.write(
            f"{command}: podman is required.\n"
            "    Install with: brew install podman (macOS) "
            "or apt install podman (Linux)\n"
        )
        sys.exit(2)
    from .podman_provisioning import ensure_image, ensure_machine_running
    try:
        ensure_machine_running(log=log)
        ensure_image(agent, log=log)
    except (FileNotFoundError, RuntimeError) as exc:
        sys.stderr.write(f"{command}: {exc}\n")
        sys.exit(2)
    return get_sandbox(
        agent=agent, env=env,
        readonly_paths=readonly_paths,
        writable_paths=writable_paths,
    )


__all__ = [
    "Sandbox",
    "get_sandbox",
    "require_sandbox_or_die",
]
