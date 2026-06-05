"""Podman container sandbox.

Each ``run()`` is a fresh ``podman run --rm`` against a pre-built
image. Host paths in ``writable_paths`` / ``readonly_paths`` are
bind-mounted into the container at the same path so the agent sees
identical paths inside and outside.

The image is *not* built here — callers must ensure it exists in the
local podman image store before constructing the sandbox.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


_PODMAN = "podman"


def build_command(
    argv: list[str],
    *,
    image: str,
    writable_paths: list[Path],
    readonly_paths: list[Path] | None = None,
    deny_network: bool = False,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
) -> list[str]:
    """Assemble a ``podman run --rm ... <image> <argv>`` invocation."""
    cmd = [_PODMAN, "run", "--rm"]
    if deny_network:
        cmd.append("--network=none")
    if cwd is not None:
        cmd.extend(["--workdir", str(cwd)])

    bound: set[str] = set()
    all_writable = list(writable_paths)
    if cwd is not None:
        all_writable.append(Path(cwd))
    for p in all_writable:
        resolved = str(Path(p).resolve())
        if resolved in bound:
            continue
        bound.add(resolved)
        cmd.extend(["--mount", f"type=bind,src={resolved},dst={resolved}"])
    for p in readonly_paths or []:
        resolved = str(Path(p).resolve())
        if resolved in bound:
            continue
        bound.add(resolved)
        cmd.extend(["--mount", f"type=bind,src={resolved},dst={resolved},ro"])

    for k, v in (env or {}).items():
        cmd.extend(["-e", f"{k}={v}"])

    cmd.append(image)
    cmd.extend(argv)
    return cmd


class PodmanSandbox:
    """Image-based sandbox using ``podman run --rm``.

    The image holds the agent CLI + deps; nothing on the host is
    visible inside the container except paths the driver explicitly
    passes via ``writable_paths`` / ``readonly_paths``.
    """

    name = "podman"

    def __init__(
        self,
        image: str,
        *,
        env: dict[str, str] | None = None,
        readonly_paths: list[Path] | None = None,
        writable_paths: list[Path] | None = None,
    ) -> None:
        self.image = image
        self._extra_env: dict[str, str] = dict(env or {})
        self._readonly_paths: list[Path] = list(readonly_paths or [])
        self._writable_paths: list[Path] = list(writable_paths or [])

    def close(self) -> None:
        """No-op. Each ``run()`` produces an ephemeral container."""

    def __enter__(self) -> "PodmanSandbox":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def wrap(
        self,
        argv: list[str],
        *,
        writable_paths: list[Path],
        deny_network: bool = False,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> list[str]:
        full_env = dict(self._extra_env)
        if env:
            full_env.update(env)
        return build_command(
            argv,
            image=self.image,
            writable_paths=list(writable_paths) + self._writable_paths,
            readonly_paths=self._readonly_paths,
            deny_network=deny_network,
            env=full_env,
            cwd=cwd,
        )

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
        wrapped = self.wrap(
            argv,
            writable_paths=writable_paths or [],
            deny_network=deny_network,
            env=env,
            cwd=cwd,
        )
        # ``--rm`` only fires on a clean container exit; if the orchestrator
        # is killed, times out, or the parent process dies before the
        # container does, the container is orphaned and its overlay layer
        # accumulates in the podman VM. Tag every invocation with a unique
        # ``--name`` so a ``finally`` can force-remove it no matter how
        # ``subprocess.run`` returned.
        import uuid
        name = f"agentcap-{uuid.uuid4().hex[:12]}"
        wrapped.insert(2, "--name")
        wrapped.insert(3, name)
        try:
            return subprocess.run(
                wrapped,
                stdin=subprocess.DEVNULL,
                capture_output=True, text=True,
                timeout=timeout, check=check,
            )
        finally:
            subprocess.run(
                [_PODMAN, "rm", "-f", name],
                stdin=subprocess.DEVNULL,
                capture_output=True, text=True,
                timeout=30,
            )

    @staticmethod
    def _runs_dir() -> Path:
        d = Path.home() / ".cache" / "agentcap" / "runs"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def mkdtemp(self, prefix: str = "agentcap-") -> str:
        return tempfile.mkdtemp(prefix=prefix, dir=str(self._runs_dir()))

    def rmtree(self, path: str) -> None:
        shutil.rmtree(path, ignore_errors=True)

    def write_text(self, path: str, content: str) -> None:
        Path(path).write_text(content)

    def read_text(self, path: str) -> str:
        return Path(path).read_text()


__all__ = ["PodmanSandbox", "build_command"]
