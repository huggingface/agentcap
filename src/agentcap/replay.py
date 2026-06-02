"""Replay primitive: resolve a captured request by id, hand back the body.

Replay is byte-faithful. No normalisation, no flags that mutate the body —
consumers that hit cross-server strictness do their own normalisation
(see AGENTS.md #3).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def load_request(source: str, request_id: str) -> dict:
    """Return the raw captured request body for ``request_id``.

    ``source`` resolves any of:
      - a local capture dir (``<rid>.request.json`` files),
      - a local ``.parquet`` produced by ``agentcap export``,
      - ``hf://datasets/<owner>/<name>`` or the bare ``<owner>/<name>``
        form.

    Raises ``KeyError`` if the id is not found.
    """
    return load_requests(source, [request_id])[request_id]


def load_requests(
    source: str, request_ids: Iterable[str]
) -> dict[str, dict]:
    """Batch form: one pass over the source per file, returns ``{id: body}``.

    Raises ``KeyError`` listing any ids that weren't found.
    """
    wanted = set(request_ids)
    if not wanted:
        return {}

    # Resolve local paths first — an existing dir/file wins over the HF
    # heuristic, so ``runs/abc/captures`` isn't misclassified as a repo.
    p = Path(source).expanduser()
    if p.is_dir():
        bodies = _load_from_capture_dir(p, wanted)
    elif p.is_file() and p.suffix == ".parquet":
        bodies = _load_from_parquet(p, wanted)
    elif _looks_like_hf_source(source):
        bodies = _load_from_hf_dataset(source, wanted)
    else:
        raise ValueError(
            f"source must be a capture dir, a .parquet file, or an "
            f"hf://datasets/... URI — got {source!r}"
        )

    missing = wanted - set(bodies)
    if missing:
        raise KeyError(
            f"request_id(s) not found in {source!r}: {sorted(missing)}"
        )
    return bodies


def _looks_like_hf_source(source: str) -> bool:
    if source.startswith("hf://"):
        return True
    # Bare ``<owner>/<name>`` — exactly one ``/`` and no path-separator
    # prefix. Heuristic for distinguishing an HF repo from a local path.
    if source.startswith((".", "/", "~")):
        return False
    return source.count("/") == 1


def _load_from_capture_dir(
    capture_dir: Path, wanted: set[str]
) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for rid in wanted:
        path = capture_dir / f"{rid}.request.json"
        if not path.is_file():
            continue
        rec = json.loads(path.read_text())
        body = rec.get("body")
        if isinstance(body, dict):
            out[rid] = body
    return out


def _load_from_parquet(
    parquet_path: Path, wanted: set[str]
) -> dict[str, dict]:
    import pyarrow.parquet as pq

    table = pq.read_table(
        str(parquet_path),
        columns=["request_id", "request"],
        filters=[("request_id", "in", list(wanted))],
    )
    return _scan_arrow_table(table, wanted)


def _load_from_hf_dataset(
    source: str, wanted: set[str]
) -> dict[str, dict]:
    """Scan every parquet under ``data/`` in the dataset until all
    wanted ids are found (or files exhausted)."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    s = source.removeprefix("hf://datasets/").strip("/")
    parts = s.split("/")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(
            f"hf source must be <owner>/<name>, got {source!r}"
        )
    repo_id = f"{parts[0]}/{parts[1]}"
    fs = HfFileSystem()
    prefix = f"datasets/{repo_id}/data"

    out: dict[str, dict] = {}
    remaining = set(wanted)
    for entry in fs.ls(prefix, detail=True):
        if entry.get("type") != "file" or not entry["name"].endswith(".parquet"):
            continue
        with fs.open(entry["name"], "rb") as fh:
            table = pq.read_table(
                fh,
                columns=["request_id", "request"],
                filters=[("request_id", "in", list(remaining))],
            )
        found = _scan_arrow_table(table, remaining)
        out.update(found)
        remaining -= set(found)
        if not remaining:
            break
    return out


def _scan_arrow_table(table, wanted: set[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    rid_col = table.column("request_id").to_pylist()
    req_col = table.column("request").to_pylist()
    for rid, req_str in zip(rid_col, req_col):
        if rid in wanted and isinstance(req_str, str):
            out[rid] = json.loads(req_str)
    return out


class AmbiguousRequestId(Exception):
    """Raised when a short rid prefix matches more than one captured
    request — caller should ask the user to disambiguate (like
    ``git`` does)."""

    def __init__(self, prefix: str, matches: list[str]):
        self.prefix = prefix
        self.matches = matches
        super().__init__(
            f"rid prefix {prefix!r} is ambiguous ({len(matches)} matches): "
            f"{', '.join(sorted(matches)[:5])}{'…' if len(matches) > 5 else ''}"
        )


def resolve_workspace_rid(
    workspace_root: Path, request_id: str
) -> tuple[Path, str] | None:
    """Find the capture dir + full rid for a (possibly truncated) request id.

    Returns ``(capture_dir, full_rid)`` for the unique match, ``None`` if
    no match. Raises ``AmbiguousRequestId`` when multiple rids share the
    prefix.
    """
    if not workspace_root.is_dir():
        return None
    matches: list[tuple[Path, str]] = []
    for run_dir in workspace_root.iterdir():
        captures = run_dir / "captures"
        if not captures.is_dir():
            continue
        # Exact match shortcut — also makes full-length rids O(1).
        exact = captures / f"{request_id}.request.json"
        if exact.is_file():
            return captures, request_id
        for hit in captures.glob(f"{request_id}*.request.json"):
            matches.append((captures, hit.name.removesuffix(".request.json")))
    if not matches:
        return None
    if len(matches) > 1:
        raise AmbiguousRequestId(request_id, [m[1] for m in matches])
    return matches[0]
