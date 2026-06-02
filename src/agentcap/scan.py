"""Secret scan over a capture run, gating ``agentcap export``.

Shells out to `trufflehog filesystem` and parses its JSON output.
Captures and traces are scanned as plain text (JSON / JSONL); the
parquet repackaging happens after the scan, so we always check the
unpacked source.

Policy: a single ``verified`` hit aborts the export. ``unverified``
hits are reported but do not block — TruffleHog's pattern matchers
have a real false-positive rate (e.g. a 32-char alphanumeric in a
model response looks like a Box OAuth token), and we don't have
verification credentials for most providers.

Scan results are persisted to ``<run_dir>/scan.json`` so subsequent
``agentcap export`` invocations skip the (sometimes slow) verify
step. The cache is invalidated when the user passes ``--rescan`` or
when the recorded ``no_verification`` mode doesn't match the
requested mode (an unverified cache can't satisfy a verified
request).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class ScanHit:
    detector: str
    file: str
    verified: bool
    raw: str  # redacted-by-Trufflehog "Raw" field, kept for context


@dataclass
class ScanResult:
    bytes_scanned: int = 0
    chunks_scanned: int = 0
    verified: list[ScanHit] = field(default_factory=list)
    unverified: list[ScanHit] = field(default_factory=list)


class TrufflehogMissingError(RuntimeError):
    """``trufflehog`` is not on PATH (and not in ~/.local/bin)."""


_INSTALL_HINT = (
    "trufflehog is required for the pre-export secret scan but was not "
    "found on PATH. Install with:\n"
    "    curl -sSfL https://raw.githubusercontent.com/trufflesecurity/"
    "trufflehog/main/scripts/install.sh | sh -s -- -b ~/.local/bin\n"
    "Or pass --no-scan to ``agentcap export`` to skip the scan."
)


def find_trufflehog() -> str:
    """Locate the ``trufflehog`` binary. Checks PATH then
    ``~/.local/bin`` (the installer's default target).
    Raises :class:`TrufflehogMissingError` if not found."""
    on_path = shutil.which("trufflehog")
    if on_path:
        return on_path
    local = Path.home() / ".local" / "bin" / "trufflehog"
    if local.is_file() and os.access(local, os.X_OK):
        return str(local)
    raise TrufflehogMissingError(_INSTALL_HINT)


def scan_path(
    path: Path | str,
    *,
    no_verification: bool = False,
    extra_args: tuple[str, ...] = (),
) -> ScanResult:
    """Scan ``path`` (a directory or file) with trufflehog.

    ``no_verification=False`` (the default) round-trips every
    candidate against the provider's API (Stripe, AWS, GitHub, HF, …)
    so the ``verified`` bucket is high-precision. Requires network.
    Pass ``True`` for offline pattern-only matching — faster but
    everything lands as ``unverified``.
    """
    bin_path = find_trufflehog()
    argv = [
        bin_path, "filesystem", str(path),
        "--json", "--no-color",
        "--results=verified,unverified",
    ]
    if no_verification:
        argv.append("--no-verification")
    argv.extend(extra_args)

    proc = subprocess.run(
        argv, capture_output=True, text=True, check=False,
    )

    result = ScanResult()
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "DetectorName" not in rec:
            continue
        hit = ScanHit(
            detector=rec.get("DetectorName") or "?",
            file=(
                rec.get("SourceMetadata", {})
                .get("Data", {})
                .get("Filesystem", {})
                .get("file") or "?"
            ),
            verified=bool(rec.get("Verified")),
            raw=str(rec.get("Raw") or "")[:80],
        )
        (result.verified if hit.verified else result.unverified).append(hit)

    # The summary line on stderr looks like:
    #   ... finished scanning {"chunks":..., "bytes":..., "verified_secrets":..., "unverified_secrets":...}
    # Parse what we can; the per-hit list above is authoritative.
    for line in proc.stderr.splitlines():
        if "finished scanning" not in line:
            continue
        brace = line.find("{")
        if brace < 0:
            continue
        try:
            stats = json.loads(line[brace:])
        except json.JSONDecodeError:
            continue
        result.bytes_scanned = int(stats.get("bytes", 0))
        result.chunks_scanned = int(stats.get("chunks", 0))
        break

    return result


SCAN_CACHE_NAME = "scan.json"


def _result_to_dict(result: ScanResult, *, no_verification: bool) -> dict:
    return {
        "scanned_at": int(time.time()),
        "no_verification": no_verification,
        "bytes_scanned": result.bytes_scanned,
        "chunks_scanned": result.chunks_scanned,
        "verified": [asdict(h) for h in result.verified],
        "unverified": [asdict(h) for h in result.unverified],
    }


def _result_from_dict(d: dict) -> ScanResult:
    return ScanResult(
        bytes_scanned=int(d.get("bytes_scanned") or 0),
        chunks_scanned=int(d.get("chunks_scanned") or 0),
        verified=[ScanHit(**h) for h in (d.get("verified") or [])],
        unverified=[ScanHit(**h) for h in (d.get("unverified") or [])],
    )


def load_cached_scan(
    run_dir: Path | str, *, no_verification: bool,
) -> ScanResult | None:
    """Return a previously persisted scan if it covers the requested
    verification mode. A cache produced with ``no_verification=True``
    cannot satisfy a ``no_verification=False`` request (the verified
    bucket would be unsound), so we re-scan in that direction.
    Returns ``None`` when no usable cache exists."""
    cache_path = Path(run_dir) / SCAN_CACHE_NAME
    if not cache_path.is_file():
        return None
    try:
        d = json.loads(cache_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    cached_no_verify = bool(d.get("no_verification", True))
    if cached_no_verify and not no_verification:
        # Want verified results; cache only has patterns.
        return None
    return _result_from_dict(d)


_SCAN_SUBDIRS = ("captures", "traces", "sessions")


def scan_run_dir(
    run_dir: Path | str,
    *,
    no_verification: bool = False,
    rescan: bool = False,
) -> tuple[ScanResult, bool]:
    """Scan a run dir, persisting the result to ``<run_dir>/scan.json``
    for cheap reuse. Returns ``(result, was_cached)``.

    Scans the three subdirs that can hold user/agent text — captures,
    traces, and sessions — and skips top-level files like
    ``run.json`` and the cache itself. ``rescan=True`` ignores any
    persisted result and re-runs trufflehog. Otherwise the cache is
    used when it covers the requested mode."""
    run_dir = Path(run_dir)
    if not rescan:
        cached = load_cached_scan(run_dir, no_verification=no_verification)
        if cached is not None:
            return cached, True

    merged = ScanResult()
    for name in _SCAN_SUBDIRS:
        sub = run_dir / name
        if not sub.is_dir():
            continue
        part = scan_path(sub, no_verification=no_verification)
        merged.bytes_scanned += part.bytes_scanned
        merged.chunks_scanned += part.chunks_scanned
        merged.verified.extend(part.verified)
        merged.unverified.extend(part.unverified)

    try:
        (run_dir / SCAN_CACHE_NAME).write_text(
            json.dumps(
                _result_to_dict(merged, no_verification=no_verification),
                indent=2,
            )
        )
    except OSError:
        # Cache write isn't load-bearing — let the scan result through.
        pass
    return merged, False


__all__ = [
    "SCAN_CACHE_NAME",
    "ScanHit",
    "ScanResult",
    "TrufflehogMissingError",
    "find_trufflehog",
    "load_cached_scan",
    "scan_path",
    "scan_run_dir",
]
