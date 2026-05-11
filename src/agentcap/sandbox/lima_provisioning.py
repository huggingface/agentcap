"""Per-agent Lima VM lifecycle: ``ensure_vm`` for ``agentcap run``
and the pytest fixture both. One source of truth for "is this VM
the one our current template would produce, and if not, rebuild
it."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


_TEMPLATE_DIR = (
    Path(__file__).resolve().parents[3] / "scripts" / "lima"
)


def template_path(agent: str) -> Path:
    return _TEMPLATE_DIR / f"agentcap-{agent}.yaml"


def _template_hash(template: Path) -> str:
    return hashlib.sha256(template.read_bytes()).hexdigest()


def _hash_file(vm: str) -> Path:
    # Lives next to Lima's own VM state.
    return Path.home() / ".lima" / vm / "agentcap-template.sha256"


def _vm_info(vm: str) -> dict | None:
    """Return the Lima list-record for ``vm`` or ``None`` if absent."""
    if not shutil.which("limactl"):
        return None
    r = subprocess.run(
        ["limactl", "list", "--format", "json", vm],
        capture_output=True, text=True,
    )
    if r.returncode != 0 or not r.stdout.strip():
        return None
    for line in r.stdout.strip().splitlines():
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("name") == vm:
            return obj
    return None


def _vm_is_from_current_template(vm: str, template: Path) -> bool:
    """``True`` iff a hash file exists *and* matches the current
    template. Missing-or-different hash → treat as stale."""
    hf = _hash_file(vm)
    if not hf.is_file():
        return False
    return hf.read_text().strip() == _template_hash(template)


def ensure_vm(
    agent: str,
    *,
    log: callable = lambda msg: None,
) -> str:
    """Bring the ``agentcap-<agent>`` VM to a Running state from the
    current ``scripts/lima/agentcap-<agent>.yaml`` template; return
    the VM name. Recreates stale VMs (delete + create) so the
    caller never has to manage the lifecycle by hand.

    ``log`` is a callable that takes one string — used for progress
    output. Defaults to silent; agentcap run wires stderr, the
    test fixture wires its ``_log`` helper.

    Raises ``FileNotFoundError`` if the template is missing,
    ``RuntimeError`` if ``limactl`` isn't installed or the start
    fails terminally.
    """
    if not shutil.which("limactl"):
        raise RuntimeError("limactl not on $PATH (brew install lima)")
    vm = f"agentcap-{agent}"
    template = template_path(agent)
    if not template.is_file():
        raise FileNotFoundError(f"template not found: {template}")

    info = _vm_info(vm)
    status = info.get("status") if info else None

    if status is not None and not _vm_is_from_current_template(vm, template):
        log(f"{vm} is stale (template hash mismatch or missing); recreating…")
        subprocess.run(
            ["limactl", "stop", "--force", vm],
            capture_output=True, text=True, timeout=120,
        )
        subprocess.run(
            ["limactl", "delete", "--force", vm],
            capture_output=True, text=True, timeout=60,
        )
        status = None

    if status is None:
        log(f"creating + starting {vm} (cold boot can take 30s+)…")
        r = subprocess.run(
            ["limactl", "start", f"--name={vm}", "--tty=false", str(template)],
            capture_output=True, text=True, timeout=900,
        )
        if r.returncode == 0:
            _hash_file(vm).write_text(_template_hash(template))
    elif status != "Running":
        log(f"starting {vm} (was {status})…")
        subprocess.run(
            ["limactl", "start", "--tty=false", vm],
            capture_output=True, text=True, timeout=300,
        )

    # Trust post-start status, not limactl's exit code.
    final_info = _vm_info(vm)
    final = final_info.get("status") if final_info else None
    if final != "Running":
        raise RuntimeError(
            f"{vm!r} is not Running after start (status={final!r})"
        )
    log(f"{vm} ready")
    return vm


def stop_vm(vm: str) -> None:
    subprocess.run(
        ["limactl", "stop", vm],
        capture_output=True, text=True, timeout=120,
    )


__all__ = [
    "ensure_vm",
    "stop_vm",
    "template_path",
]
