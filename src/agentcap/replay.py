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
      - ``hf://datasets/<owner>/<name>[/<subdir>]`` or the bare
        ``<owner>/<name>[/<subdir>]`` form.

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

    if _looks_like_hf_source(source):
        bodies = _load_from_hf_dataset(source, wanted)
    else:
        p = Path(source)
        if p.is_dir():
            bodies = _load_from_capture_dir(p, wanted)
        elif p.is_file() and p.suffix == ".parquet":
            bodies = _load_from_parquet(p, wanted)
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
    # Bare ``<owner>/<name>[/<subdir>]`` — a single ``/`` and no path
    # separator-style prefix. Heuristic, mirrored from how
    # ``export.parse_dataset_uri`` accepts the same form.
    if source.startswith((".", "/", "~")):
        return False
    return "/" in source


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
        str(parquet_path), columns=["request_id", "request"]
    )
    return _scan_arrow_table(table, wanted)


def _load_from_hf_dataset(
    source: str, wanted: set[str]
) -> dict[str, dict]:
    """Scan every parquet under ``data/[<subdir>/]`` in the dataset
    until all wanted ids are found (or files exhausted)."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    from .export import parse_dataset_uri

    repo_id, subdir = parse_dataset_uri(source)
    fs = HfFileSystem()
    prefix = f"datasets/{repo_id}/data" + (f"/{subdir}" if subdir else "")

    out: dict[str, dict] = {}
    remaining = set(wanted)
    for entry in fs.ls(prefix, detail=True):
        if entry.get("type") != "file" or not entry["name"].endswith(".parquet"):
            continue
        with fs.open(entry["name"], "rb") as fh:
            table = pq.read_table(fh, columns=["request_id", "request"])
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


def resolve_workspace_rid(
    workspace_root: Path, request_id: str
) -> Path | None:
    """Find the capture dir containing ``<request_id>.request.json`` by
    scanning ``<workspace_root>/*/captures/``. Returns the capture dir
    path, or ``None`` if not found."""
    if not workspace_root.is_dir():
        return None
    for run_dir in workspace_root.iterdir():
        captures = run_dir / "captures"
        if (captures / f"{request_id}.request.json").is_file():
            return captures
    return None
