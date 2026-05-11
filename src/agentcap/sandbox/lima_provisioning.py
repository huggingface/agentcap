"""Per-agent Lima VM lifecycle: ``ensure_vm`` for ``agentcap run``
and the pytest fixture both. One source of truth for "is this VM
the one our current template would produce, and if not, rebuild
it." After (re)starting the VM, the per-agent bundle from
``containers/agentcap-<agent>/`` is pushed in — Lima equivalent of
the Containerfile's ``COPY`` directives."""

from __future__ import annotations

import base64
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


_TEMPLATE_DIR = (
    Path(__file__).resolve().parents[3] / "scripts" / "lima"
)
_CONTAINERS_DIR = (
    Path(__file__).resolve().parents[3] / "containers"
)
_INIT_PATH = "/usr/local/bin/agentcap-init"

# Per-agent VM-side paths for the templated config files baked into
# ``containers/agentcap-<agent>/``. Mirrors the Containerfile
# ``COPY`` directives. ``$HOME`` is expanded by the VM shell.
_BUNDLE_FILES: dict[str, list[tuple[str, str]]] = {
    "opencode": [
        ("opencode.json", "$HOME/.config/opencode/opencode.json"),
    ],
    "pi": [
        # /opt/pi-config is created + chown'd to the Lima user by
        # the agentcap-pi.yaml provision, so this push lands as the
        # regular user.
        ("models.json", "/opt/pi-config/models.json"),
    ],
    # hermes has no templated config file — agentcap-init runs
    # ``hermes config set`` directly. No bundle entry needed.
}


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


def _vm_shell(vm: str, shell_cmd: str, *, timeout: float = 60) -> None:
    """Run a single shell command inside ``vm``; raise on failure."""
    subprocess.run(
        ["limactl", "shell", vm, "--", "sh", "-c", shell_cmd],
        capture_output=True, text=True, check=True, timeout=timeout,
    )


def _push_root_file(vm: str, content: bytes, dst: str, mode: int) -> None:
    """Install ``content`` at root-owned ``dst`` (e.g. /usr/local/bin/…)
    via sudo + base64-pipe to bypass shell quoting on the payload."""
    encoded = base64.b64encode(content).decode("ascii")
    parent = dst.rsplit("/", 1)[0]
    _vm_shell(
        vm,
        f"sudo mkdir -p '{parent}' && "
        f"printf %s '{encoded}' | base64 -d | sudo tee '{dst}' >/dev/null && "
        f"sudo chmod {mode:o} '{dst}'",
    )


def _push_user_file(vm: str, content: bytes, dst: str) -> None:
    """Install ``content`` at a user-relative ``dst`` (may contain
    ``$HOME``). Parent dir is mkdir -p'd."""
    encoded = base64.b64encode(content).decode("ascii")
    _vm_shell(
        vm,
        f'mkdir -p "$(dirname {dst})" && '
        f"printf %s '{encoded}' | base64 -d > {dst}",
    )


def _push_bundle(
    vm: str,
    agent: str,
    log: callable = lambda msg: None,
) -> None:
    """Mirror ``containers/agentcap-<agent>/*`` into the running VM
    at the same canonical paths the Containerfile installs them.
    Idempotent; safe to re-run."""
    bundle = _CONTAINERS_DIR / f"agentcap-{agent}"
    if not bundle.is_dir():
        return
    init_sh = bundle / "agentcap-init.sh"
    cfg_map = _BUNDLE_FILES.get(agent, [])
    if not init_sh.is_file() and not cfg_map:
        return
    log(f"installing agentcap bundle into {vm}…")
    if init_sh.is_file():
        _push_root_file(vm, init_sh.read_bytes(), _INIT_PATH, mode=0o755)
    for src_name, dst in cfg_map:
        src = bundle / src_name
        if src.is_file():
            _push_user_file(vm, src.read_bytes(), dst)


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
    _push_bundle(vm, agent, log=log)
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
