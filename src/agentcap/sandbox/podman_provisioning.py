"""Per-agent podman image lifecycle: ``ensure_image`` for
``agentcap run`` and the pytest fixture both.

The Containerfile is the source of truth: its SHA256 is baked into
the built image as a label, and a hash mismatch on subsequent runs
forces a rebuild.
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
from pathlib import Path

_CONTAINERFILE_DIR = (
    Path(__file__).resolve().parents[3] / "containers"
)

_HASH_LABEL = "agentcap.containerfile-hash"


def containerfile_path(agent: str) -> Path:
    return _CONTAINERFILE_DIR / f"agentcap-{agent}.Containerfile"


def image_tag(agent: str) -> str:
    return f"localhost/agentcap-{agent}:latest"


def _containerfile_hash(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    name = path.stem
    ctx = path.parent / name
    if ctx.is_dir():
        for f in sorted(ctx.rglob("*")):
            if f.is_file():
                h.update(str(f.relative_to(ctx)).encode())
                h.update(b"\0")
                h.update(f.read_bytes())
                h.update(b"\0")
    return h.hexdigest()


def _image_info(tag: str) -> dict | None:
    if not shutil.which("podman"):
        return None
    r = subprocess.run(
        ["podman", "image", "inspect", tag],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        return None
    try:
        info = json.loads(r.stdout)
    except json.JSONDecodeError:
        return None
    return info[0] if isinstance(info, list) and info else None


def _image_stored_hash(info: dict) -> str | None:
    labels = (info.get("Labels") or info.get("Config", {}).get("Labels")) or {}
    return labels.get(_HASH_LABEL)


def _image_is_current(tag: str, cf: Path) -> bool:
    info = _image_info(tag)
    if info is None:
        return False
    stored = _image_stored_hash(info)
    return stored is not None and stored == _containerfile_hash(cf)


def ensure_image(
    agent: str,
    *,
    log=lambda msg: None,
) -> str:
    """Build the per-agent podman image from the Containerfile if
    absent or stale; return the image tag.

    Raises ``FileNotFoundError`` if the Containerfile is missing,
    ``RuntimeError`` if ``podman`` isn't installed or the build fails.
    """
    if not shutil.which("podman"):
        raise RuntimeError(
            "podman not on $PATH (brew install podman / apt install podman)"
        )
    cf = containerfile_path(agent)
    if not cf.is_file():
        raise FileNotFoundError(f"Containerfile not found: {cf}")
    tag = image_tag(agent)

    if _image_is_current(tag, cf):
        log(f"{tag} ready (Containerfile hash match)")
        return tag

    if _image_info(tag) is not None:
        log(f"{tag} is stale; rebuilding…")
        subprocess.run(
            ["podman", "rmi", "--force", tag],
            capture_output=True, text=True, check=False,
        )
    else:
        log(f"{tag} not built; building (cold build can take minutes)…")

    cf_hash = _containerfile_hash(cf)
    r = subprocess.run(
        [
            "podman", "build",
            "-f", str(cf),
            "-t", tag,
            "--label", f"{_HASH_LABEL}={cf_hash}",
            str(cf.parent),
        ],
        timeout=1800,
    )
    if r.returncode != 0:
        raise RuntimeError(
            f"podman build failed for {tag} (rc={r.returncode}); "
            f"see streamed output above."
        )
    log(f"{tag} built")
    return tag


def rmi_image(tag: str) -> None:
    subprocess.run(
        ["podman", "rmi", "--force", tag],
        capture_output=True, text=True, timeout=60, check=False,
    )


def _machine_status() -> str | None:
    """Return the status (``Running`` / ``Stopped`` / ``Starting`` /
    ...) of the default podman machine, or ``None`` if no machine
    exists."""
    if not shutil.which("podman"):
        return None
    r = subprocess.run(
        ["podman", "machine", "list", "--format", "json"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        return None
    try:
        machines = json.loads(r.stdout)
    except json.JSONDecodeError:
        return None
    if not machines:
        return None
    default = next(
        (m for m in machines if m.get("Default")), machines[0],
    )
    if default.get("Running"):
        return "Running"
    if default.get("Starting"):
        return "Starting"
    return "Stopped"


def ensure_machine_running(*, log=lambda msg: None) -> None:
    """macOS only: ensure ``podman machine`` is up. No-op on Linux,
    where podman talks to the host kernel directly.

    Never auto-initialises the machine — that's a 1-2 GB download
    and a multi-minute operation the user should consent to. Raises
    ``RuntimeError`` if podman isn't installed, no machine exists,
    or the machine can't be started.
    """
    if platform.system() != "Darwin":
        return
    if not shutil.which("podman"):
        raise RuntimeError(
            "podman not on $PATH (brew install podman)"
        )
    status = _machine_status()
    if status is None:
        raise RuntimeError(
            "no podman machine found. Initialise one first:\n"
            "    podman machine init\n"
            "    podman machine start"
        )
    if status == "Running":
        return
    log(f"podman machine is {status}; starting…")
    r = subprocess.run(
        ["podman", "machine", "start"],
        capture_output=True, text=True, timeout=300,
    )
    if r.returncode != 0:
        raise RuntimeError(
            f"podman machine start failed (rc={r.returncode}): "
            f"{r.stderr.strip()}"
        )


__all__ = [
    "containerfile_path",
    "ensure_image",
    "ensure_machine_running",
    "image_tag",
    "rmi_image",
]
