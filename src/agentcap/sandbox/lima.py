"""Lima backend: per-agent Linux VM as the macOS sandbox.

The Mac host is reachable from inside the VM as
``host.lima.internal``. ``limactl shell`` has no ``--env`` flag, so
env is injected via the POSIX ``env KEY=VAL …`` prefix on the VM
side (see :meth:`LimaSandbox._build_shell_cmd`).

``run()`` and ``wrap()`` prepend the canonical
``/usr/local/bin/agentcap-init`` to the agent argv — mirror of the
Containerfile ``ENTRYPOINT`` so the init script can render config
files from env vars before exec'ing the agent.
:meth:`_exec_in_vm` (used by mkdtemp / write_text / …) skips the
init prefix: those are sandbox-internal ops, and they need to work
even before the bundle is installed by
:func:`lima_provisioning.ensure_vm`.
"""

from __future__ import annotations

import base64
import os
import subprocess
import sys
from pathlib import Path


_DEFAULT_VM = "agentcap-unset"
_LIMACTL = "limactl"

INIT_PATH = "/usr/local/bin/agentcap-init"


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
        env: dict[str, str] | None = None,
        readonly_paths: list[Path] | None = None,
        writable_paths: list[Path] | None = None,
    ) -> None:
        # ``env`` is the sandbox-wide baked env merged with per-call
        # env at run time. ``readonly_paths`` / ``writable_paths`` are
        # mirrored here for API parity with ``BwrapSandbox``; the
        # actual mounts are configured at VM provision time (see
        # :func:`lima_provisioning.ensure_vm`).
        self.vm = vm
        self.workdir = workdir
        self._baked_env: dict[str, str] = dict(env or {})
        self._readonly_paths: list[Path] = list(readonly_paths or [])
        self._writable_paths: list[Path] = list(writable_paths or [])

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
        # Inject the baked env via POSIX ``env`` so init scripts that
        # require AGENTCAP_MODEL etc. see them — same shape as
        # BwrapSandbox.wrap which threads ``self._baked_env`` through.
        inner = [INIT_PATH] + argv
        if self._baked_env:
            inner = (
                ["env"] + [f"{k}={v}" for k, v in self._baked_env.items()]
                + inner
            )
        return build_command(inner, vm=self.vm, workdir=self.workdir)

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
        full_env = dict(self._baked_env)
        if env:
            full_env.update(env)
        cmd = self._build_shell_cmd(
            [INIT_PATH] + list(argv), env=full_env or None, cwd=cwd,
        )
        try:
            # agentcap is headless; close stdin so any agent that
            # probes the TTY (consent prompts, ``read -p``) gets
            # immediate EOF instead of hanging.
            return subprocess.run(
                cmd,
                env=os.environ.copy(),
                stdin=subprocess.DEVNULL,
                capture_output=True, text=True,
                timeout=timeout, check=check,
            )
        except subprocess.TimeoutExpired:
            # Python's SIGKILL went to the local ``limactl shell``
            # wrapper; the inner agent process inside the VM has been
            # orphaned (sshd's session-close doesn't propagate cleanly
            # to non-tty children) and will keep firing requests at the
            # capture proxy until it finishes its loop. Read the PID
            # ``agentcap-init`` wrote just before ``exec``ing the agent
            # and kill exactly that process before re-raising.
            self._kill_current_pid_in_vm()
            raise

    def _kill_current_pid_in_vm(self) -> None:
        """Read ``/tmp/agentcap-current.pid`` (written by
        ``agentcap-init`` just before the final ``exec``) and SIGKILL
        that PID inside the VM. Best-effort: short timeout, never
        raises — a missing file or already-dead process is fine."""
        try:
            r = self._exec_in_vm(
                ["cat", "/tmp/agentcap-current.pid"],
                check=False, timeout=5,
            )
            pid = (r.stdout or "").strip()
            if not pid or not pid.isdigit():
                return
            self._exec_in_vm(
                ["kill", "-9", pid], check=False, timeout=5,
            )
        except Exception:
            pass

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


__all__ = ["LimaSandbox", "build_command", "INIT_PATH"]
