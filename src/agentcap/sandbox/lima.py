"""Lima backend: per-agent Linux VM as the macOS sandbox.

The Mac host is reachable from inside the VM as
``host.lima.internal``. ``limactl shell`` has no ``--env`` flag, so
env is injected via the POSIX ``env KEY=VAL …`` prefix on the VM
side (see :meth:`LimaSandbox._build_shell_cmd`).
"""

from __future__ import annotations

import base64
import os
import subprocess
import sys
from pathlib import Path


_DEFAULT_VM = "agentcap-unset"
_LIMACTL = "limactl"


def build_command(
    argv: list[str],
    *,
    vm: str = _DEFAULT_VM,
    workdir: Path | str | None = None,
    limactl: str = _LIMACTL,
) -> list[str]:
    """Build a ``limactl shell ... -- argv`` invocation. Public for testing."""
    cmd = [limactl, "shell"]
    if workdir is not None:
        cmd.extend(["--workdir", str(workdir)])
    cmd.append(vm)
    cmd.append("--")
    cmd.extend(argv)
    return cmd


class LimaSandbox:
    name = "lima"
    binary = _LIMACTL

    def __init__(
        self,
        vm: str = _DEFAULT_VM,
        workdir: Path | str | None = None,
    ) -> None:
        self.vm = vm
        self.workdir = workdir

    def wrap(
        self,
        argv: list[str],
        *,
        writable_paths: list[Path],
        deny_network: bool = False,
    ) -> list[str]:
        if deny_network:
            sys.stderr.write(
                "[lima-sandbox] WARN deny_network=True ignored; "
                "Lima does not support per-call network policy. "
                "Use BwrapSandbox for real network isolation.\n"
            )
        # writable_paths is advisory on Lima — VM mounts decide.
        return build_command(argv, vm=self.vm, workdir=self.workdir)

    def _build_shell_cmd(
        self,
        argv: list[str],
        *,
        env: dict[str, str] | None,
        cwd: str | None,
    ) -> list[str]:
        # limactl shell has no --env flag; prepend POSIX `env K=V …`
        # on the VM side. Separate argv items avoid shell quoting.
        cmd = [self.binary, "shell"]
        if cwd is not None:
            cmd.extend(["--workdir", str(cwd)])
        cmd.append(self.vm)
        cmd.append("--")
        if env:
            cmd.append("env")
            for k, v in env.items():
                cmd.append(f"{k}={v}")
        cmd.extend(argv)
        return cmd

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
        if deny_network:
            sys.stderr.write(
                "[lima-sandbox] WARN deny_network=True ignored on Lima.\n"
            )
        cmd = self._build_shell_cmd(list(argv), env=env, cwd=cwd)
        return subprocess.run(
            cmd,
            env=os.environ.copy(),
            capture_output=True, text=True,
            timeout=timeout, check=check,
        )

    def _exec_in_vm(
        self,
        argv: list[str],
        *,
        check: bool = True,
        timeout: float | None = 60,
    ) -> subprocess.CompletedProcess:
        cmd = self._build_shell_cmd(argv, env=None, cwd=None)
        return subprocess.run(
            cmd,
            capture_output=True, text=True,
            timeout=timeout, check=check,
        )

    def mkdtemp(self, prefix: str = "agentcap-") -> str:
        # mktemp template needs at least 3 X's.
        r = self._exec_in_vm(["mktemp", "-d", f"/tmp/{prefix}XXXXXXXX"])
        return r.stdout.strip()

    def rmtree(self, path: str) -> None:
        self._exec_in_vm(["rm", "-rf", "--", path], check=False)

    def write_text(self, path: str, content: str) -> None:
        # base64-pipe to bypass shell quoting on the content.
        encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
        shell_cmd = (
            f"printf %s '{encoded}' | base64 -d > {_shell_quote(path)}"
        )
        self._exec_in_vm(["sh", "-c", shell_cmd])

    def read_text(self, path: str) -> str:
        r = self._exec_in_vm(["cat", "--", path])
        return r.stdout


def _shell_quote(s: str) -> str:
    """POSIX single-quoting; embedded `'` becomes `'\\''`."""
    return "'" + s.replace("'", "'\\''") + "'"


__all__ = ["LimaSandbox", "build_command"]
