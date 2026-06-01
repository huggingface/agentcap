"""Unit tests for ``agentcap.scan``.

These exercise the real trufflehog binary (skipped when not
installed). Cache behaviour is checked by calling ``scan_run_dir``
twice and inspecting the returned ``was_cached`` flag + the
persisted ``scan.json`` — no monkeypatching of ``scan_path`` itself,
so the tests fail if the cache short-circuit regresses.

Missing-binary errors are exercised by manipulating ``PATH`` /
``HOME`` (legitimate inputs to ``find_trufflehog``), not by
mocking ``shutil.which``.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from agentcap.scan import (
    SCAN_CACHE_NAME,
    TrufflehogMissingError,
    find_trufflehog,
    load_cached_scan,
    scan_path,
    scan_run_dir,
)


def _has_trufflehog() -> bool:
    if shutil.which("trufflehog"):
        return True
    local = Path.home() / ".local" / "bin" / "trufflehog"
    return local.is_file() and os.access(local, os.X_OK)


_HAS_TRUFFLEHOG = _has_trufflehog()


live = pytest.mark.skipif(
    not _HAS_TRUFFLEHOG, reason="trufflehog binary not installed"
)


def _make_run_dir(root: Path, *, with_poisoned_capture: bool = False) -> Path:
    """Minimal run layout with only ``captures/`` populated, so each
    test controls exactly which subdir trufflehog has to scan."""
    run_dir = root / "agent-local-20260601-000000"
    captures = run_dir / "captures"
    captures.mkdir(parents=True)
    body = '{"request_id":"rid","captured_at":1,"body":{"model":"m","messages":[]}}'
    (captures / "rid.request.json").write_text(body)
    if with_poisoned_capture:
        # Stripe doc-example key — pattern-matches the Stripe detector,
        # never verifies against the live API.
        (captures / "poisoned.request.json").write_text(
            '{"messages":[{"role":"user","content":"sk_live_4eC39HqLyjWDarjtT1zdp7dc"}]}'
        )
    return run_dir


# ---------------------------------------------------------------------------
# find_trufflehog — PATH / HOME-driven, no mocking
# ---------------------------------------------------------------------------


def test_find_trufflehog_raises_when_path_and_home_are_empty(
    monkeypatch, tmp_path,
):
    """Both PATH lookup and ~/.local/bin fallback miss → install hint."""
    monkeypatch.setenv("PATH", str(tmp_path))  # nothing in this dir
    monkeypatch.setenv("HOME", str(tmp_path))  # no .local/bin/trufflehog under HOME
    with pytest.raises(TrufflehogMissingError) as exc:
        find_trufflehog()
    assert "install.sh" in str(exc.value)


def test_find_trufflehog_falls_back_to_local_bin(monkeypatch, tmp_path):
    """No PATH hit → ~/.local/bin/trufflehog wins."""
    fake = tmp_path / ".local" / "bin" / "trufflehog"
    fake.parent.mkdir(parents=True)
    fake.write_text("#!/bin/sh\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))  # empty
    monkeypatch.setenv("HOME", str(tmp_path))
    assert find_trufflehog() == str(fake)


@live
def test_find_trufflehog_finds_installed_binary():
    """The real installed binary is locatable."""
    bin_path = find_trufflehog()
    assert os.path.basename(bin_path) == "trufflehog"
    assert os.access(bin_path, os.X_OK)


# ---------------------------------------------------------------------------
# scan_path — runs the real binary
# ---------------------------------------------------------------------------


@live
def test_scan_path_clean_dir_no_hits(tmp_path):
    (tmp_path / "f.json").write_text('{"model": "m", "messages": []}\n')
    result = scan_path(tmp_path, no_verification=True)
    assert result.verified == []
    assert result.unverified == []
    assert result.chunks_scanned >= 1


@live
def test_scan_path_detects_unverified_stripe_pattern(tmp_path):
    """A canned Stripe-shaped string trips the Stripe detector.
    With ``no_verification=True`` Stripe's docs example lands as
    unverified — we don't call the live API."""
    (tmp_path / "poisoned.json").write_text(
        '{"messages":[{"role":"user","content":"sk_live_4eC39HqLyjWDarjtT1zdp7dc"}]}'
    )
    result = scan_path(tmp_path, no_verification=True)
    assert result.verified == []
    assert len(result.unverified) >= 1
    assert result.unverified[0].detector.lower() == "stripe"
    assert result.unverified[0].file.endswith("poisoned.json")


# ---------------------------------------------------------------------------
# scan_run_dir — cache write + reuse + mode mismatch (real binary)
# ---------------------------------------------------------------------------


@live
def test_scan_run_dir_writes_cache_on_first_call(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    result, was_cached = scan_run_dir(run_dir, no_verification=True)
    assert was_cached is False
    cache_path = run_dir / SCAN_CACHE_NAME
    assert cache_path.is_file(), "scan.json should be written on first call"
    cache = json.loads(cache_path.read_text())
    assert cache["no_verification"] is True
    assert cache["chunks_scanned"] == result.chunks_scanned
    assert cache["bytes_scanned"] == result.bytes_scanned


@live
def test_scan_run_dir_reuses_cache_on_second_call(tmp_path):
    """Second call in the same mode short-circuits — no trufflehog
    invocation. Verified by stat-ing the cache mtime: a fresh scan
    would rewrite it."""
    run_dir = _make_run_dir(tmp_path)
    scan_run_dir(run_dir, no_verification=True)
    mtime_after_first = (run_dir / SCAN_CACHE_NAME).stat().st_mtime

    _, was_cached = scan_run_dir(run_dir, no_verification=True)
    assert was_cached is True
    assert (run_dir / SCAN_CACHE_NAME).stat().st_mtime == mtime_after_first


@live
def test_scan_run_dir_rescan_forces_fresh_subprocess(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    scan_run_dir(run_dir, no_verification=True)

    # mtime resolution on linux can be coarse — overwrite the cache
    # with a sentinel so a successful re-scan is observable by the
    # cache content changing back to a real ScanResult.
    (run_dir / SCAN_CACHE_NAME).write_text("{}")

    _, was_cached = scan_run_dir(run_dir, no_verification=True, rescan=True)
    assert was_cached is False
    # Cache rewritten with a real ScanResult, not the sentinel ``{}``.
    refreshed = json.loads((run_dir / SCAN_CACHE_NAME).read_text())
    assert "chunks_scanned" in refreshed
    assert refreshed["chunks_scanned"] >= 1


@live
def test_scan_run_dir_rescans_when_verify_request_meets_pattern_only_cache(
    tmp_path,
):
    """Pattern-only cache can't satisfy a verified request — re-scan."""
    run_dir = _make_run_dir(tmp_path)
    scan_run_dir(run_dir, no_verification=True)  # pattern-only cache
    _, was_cached = scan_run_dir(run_dir, no_verification=False)
    assert was_cached is False
    # Cache file now records the new (verified) mode.
    cache = json.loads((run_dir / SCAN_CACHE_NAME).read_text())
    assert cache["no_verification"] is False


@live
def test_scan_run_dir_verified_cache_satisfies_pattern_request(tmp_path):
    """Verified cache covers a pattern-only request (patterns are a
    subset of what verification ran on)."""
    run_dir = _make_run_dir(tmp_path)
    scan_run_dir(run_dir, no_verification=False)  # verified cache
    _, was_cached = scan_run_dir(run_dir, no_verification=True)
    assert was_cached is True


@live
def test_scan_run_dir_finds_hits_in_captures(tmp_path):
    run_dir = _make_run_dir(tmp_path, with_poisoned_capture=True)
    result, _ = scan_run_dir(run_dir, no_verification=True)
    assert result.verified == []
    assert any(h.detector.lower() == "stripe" for h in result.unverified)


@live
def test_scan_run_dir_excludes_self_cache(tmp_path):
    """scan.json itself must never be in the scan corpus — that would
    inflate chunk counts on every re-scan."""
    run_dir = _make_run_dir(tmp_path)
    first, _ = scan_run_dir(run_dir, no_verification=True)
    second, _ = scan_run_dir(run_dir, no_verification=True, rescan=True)
    assert first.chunks_scanned == second.chunks_scanned
    assert first.bytes_scanned == second.bytes_scanned


# ---------------------------------------------------------------------------
# load_cached_scan — pure file IO, no binary needed
# ---------------------------------------------------------------------------


def test_load_cached_scan_returns_none_when_missing(tmp_path):
    assert load_cached_scan(tmp_path, no_verification=False) is None


def test_load_cached_scan_returns_none_on_invalid_json(tmp_path):
    (tmp_path / SCAN_CACHE_NAME).write_text("not json at all")
    assert load_cached_scan(tmp_path, no_verification=False) is None


def test_load_cached_scan_rejects_pattern_only_when_verify_requested(
    tmp_path,
):
    (tmp_path / SCAN_CACHE_NAME).write_text(json.dumps({
        "scanned_at": 0,
        "no_verification": True,
        "bytes_scanned": 0,
        "chunks_scanned": 0,
        "verified": [],
        "unverified": [],
    }))
    # Pattern-only cache; caller wants verified → cache is not enough.
    assert load_cached_scan(tmp_path, no_verification=False) is None
    # Same cache satisfies a pattern-only request.
    result = load_cached_scan(tmp_path, no_verification=True)
    assert result is not None
