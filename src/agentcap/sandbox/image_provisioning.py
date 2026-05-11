"""Per-agent buildah image lifecycle: ``ensure_image`` for
``agentcap run`` and the pytest fixture both. Mirror of
``lima_provisioning.py`` for the Linux/bwrap path.

The Containerfile is the source of truth: its SHA256 is baked into
the built image as a label, and a hash mismatch on subsequent runs
forces a rebuild. Same staleness model as the Lima per-agent VM.
"""

from __future__ import annotations

import hashlib
import json
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
    return f"agentcap-{agent}:latest"


def _containerfile_hash(path: Path) -> str:
    """Hash the Containerfile *and* its per-agent build-context dir
    (``containers/agentcap-<agent>/``) so changes to baked configs /
    init scripts also invalidate the image — buildah's layer cache
    would catch them, but our short-circuit check would skip the
    build entirely if it only looked at the Containerfile.
    """
    h = hashlib.sha256()
    h.update(path.read_bytes())
    # Build-context dir is ``containers/agentcap-<agent>/`` —
    # mirror the Containerfile name. Hash every file recursively, by
    # sorted path, so the result is deterministic.
    name = path.stem  # ``agentcap-<agent>.Containerfile`` → ``agentcap-<agent>``
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
    """Return the ``buildah inspect`` record or None if absent."""
    if not shutil.which("buildah"):
        return None
    r = subprocess.run(
        ["buildah", "inspect", "--type", "image", tag],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        return None
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError:
        return None


def _image_stored_hash(info: dict) -> str | None:
    cfg = info.get("OCIv1", {}).get("config", {})
    labels = cfg.get("Labels") or {}
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
    """Build the ``agentcap-<agent>:latest`` image from
    ``containers/agentcap-<agent>.Containerfile`` if absent or stale;
    return the image tag.

    ``--isolation=chroot`` keeps the build out of user namespaces so
    `agentcap run` works on hosts with restrictive AppArmor policies
    (Ubuntu 24.04's default). The runtime still needs userns for
    bwrap — that constraint is unavoidable.

    Raises ``FileNotFoundError`` if the Containerfile is missing,
    ``RuntimeError`` if ``buildah`` isn't installed or the build
    fails.
    """
    if not shutil.which("buildah"):
        raise RuntimeError(
            "buildah not on $PATH (apt install buildah)"
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
            ["buildah", "rmi", "--force", tag],
            capture_output=True, text=True, check=False,
        )
    else:
        log(f"{tag} not built; building (cold build can take minutes)…")

    cf_hash = _containerfile_hash(cf)
    # Stream output to inherited stderr — a cold image build can take
    # several minutes, and silent capture leaves the test runner
    # looking hung. Failures still surface via the non-zero exit.
    r = subprocess.run(
        [
            "buildah", "bud",
            "--isolation", "chroot",
            "-f", str(cf),
            "-t", tag,
            "--label", f"{_HASH_LABEL}={cf_hash}",
            str(cf.parent),
        ],
        timeout=1800,
    )
    if r.returncode != 0:
        raise RuntimeError(
            f"buildah bud failed for {tag} (rc={r.returncode}); "
            f"see streamed output above."
        )
    log(f"{tag} built")
    return tag


def rmi_image(tag: str) -> None:
    subprocess.run(
        ["buildah", "rmi", "--force", tag],
        capture_output=True, text=True, timeout=60, check=False,
    )


__all__ = [
    "containerfile_path",
    "ensure_image",
    "image_tag",
    "rmi_image",
]
