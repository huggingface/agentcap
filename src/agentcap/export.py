"""Trace dir → parquet export.

For each ``<request_id>.request.json``, pair with the matching
``<request_id>.response.json`` and emit one parquet row.

Destinations: ``--output FILE.parquet`` (local) or
``--push hf://buckets/owner/name/prefix/`` (Storage Bucket;
append-by-prefix). Dataset repos are not a destination — they're
replace-only on push; render locally and ``hf upload`` instead.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from .provider import _hostname_fallback, refine_for_sub_provider


_BUCKET_PREFIX = "hf://buckets/"


def detect_provider_columns(capture_dir: Path | str) -> dict:
    """Derive ``provider`` + ``upstream_url`` from the per-request
    ``upstream_url`` stamp. Empty dict for legacy capture dirs missing
    the stamp."""
    for req_path in sorted(Path(capture_dir).glob("*.request.json")):
        try:
            rec = json.loads(req_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        upstream_url = rec.get("upstream_url")
        if not isinstance(upstream_url, str) or not upstream_url:
            continue
        model = (rec.get("body") or {}).get("model")
        provider = refine_for_sub_provider(
            _hostname_fallback(upstream_url),
            model if isinstance(model, str) else None,
        )
        return {"provider": provider, "upstream_url": upstream_url}
    return {}


def detect_model(capture_dir: Path | str) -> str | None:
    """Unique ``body.model`` across all captured requests, or ``None``.
    Raises ``ValueError`` on mixed models (datasets never mix models).
    ``@revision`` suffixes are stripped."""
    capture_dir = Path(capture_dir)
    seen: set[str] = set()
    for req_path in sorted(capture_dir.glob("*.request.json")):
        try:
            rec = json.loads(req_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        m = (rec.get("body") or {}).get("model")
        if isinstance(m, str) and m:
            seen.add(_bare_model_id(m))
    if len(seen) > 1:
        raise ValueError(
            f"capture dir contains requests for multiple models: "
            f"{sorted(seen)}. Datasets never mix models — split into "
            f"separate capture dirs and export each one independently."
        )
    return seen.pop() if seen else None


def _bare_model_id(model: str) -> str:
    """Strip ``@revision`` suffix so ``gemma-4-E4B-it`` and
    ``gemma-4-E4B-it@main`` are treated as the same id."""
    return model.split("@", 1)[0]


def _iter_pairs(
    capture_dir: Path,
) -> Iterator[tuple[str, dict, dict | None, int, dict]]:
    """Yield (request_id, request_body, response_body, captured_at,
    upstream_fingerprint) per captured request, in filename order."""
    for req_path in sorted(capture_dir.glob("*.request.json")):
        rec = json.loads(req_path.read_text())
        rid = rec.get("request_id") or req_path.stem.split(".")[0]
        captured_at = int(rec.get("captured_at", 0))
        body = rec.get("body") or {}
        resp_path = capture_dir / f"{rid}.response.json"
        resp_body: dict | None = None
        upstream_fp: dict = {}
        if resp_path.exists():
            resp_rec = json.loads(resp_path.read_text())
            upstream_fp = resp_rec.get("upstream_fingerprint") or {}
            if resp_rec.get("stream"):
                resp_body = {"stream": True, "raw": resp_rec.get("raw", "")}
            else:
                resp_body = resp_rec.get("body") or {}
        yield rid, body, resp_body, captured_at, upstream_fp


def _fingerprint_columns(fp: dict | None) -> dict:
    fp = fp or {}
    return {
        "served_by": fp.get("x_served_by"),
        "served_build_info": fp.get("build_info"),
        "served_model": fp.get("served_model"),
    }


def _row(
    request_id: str,
    request_body: dict,
    response_body: dict | None,
    captured_at: int,
    upstream_fp: dict | None,
) -> dict:
    # request / response stringified so Arrow doesn't infer a schema over
    # heterogeneous tool-schema fields. Consumers json.loads them.
    model = (request_body.get("model") or "") if isinstance(request_body, dict) else ""
    return {
        "request_id": request_id,
        "model": model,
        "captured_at": captured_at,
        "request": json.dumps(request_body, ensure_ascii=False),
        "response": json.dumps(response_body or {}, ensure_ascii=False),
        **_fingerprint_columns(upstream_fp),
    }


def export_local(
    capture_dir: Path | str,
    output: Path | str,
    *,
    batch_size: int = 32,
    progress: bool = True,
    provider_columns: dict | None = None,
) -> int:
    """Stream the capture dir into a single parquet. Returns row count.
    Batches via ``ParquetWriter`` so a mid-render kill leaves a valid
    parquet up to the last flushed batch."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    capture_dir = Path(capture_dir)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if provider_columns is None:
        provider_columns = detect_provider_columns(capture_dir)

    request_files = sorted(capture_dir.glob("*.request.json"))
    total = len(request_files)
    if total == 0:
        raise ValueError(f"no captured requests in {capture_dir}")

    pairs_iter = _iter_pairs(capture_dir)
    if progress:
        try:
            from tqdm import tqdm
            pairs_iter = tqdm(
                pairs_iter,
                total=total,
                desc=f"export {capture_dir.name}",
                unit="row",
            )
        except ImportError:
            pass

    writer: pq.ParquetWriter | None = None
    schema: pa.Schema | None = None
    batch: list[dict] = []
    n_written = 0

    def _flush(rows: list[dict]) -> None:
        nonlocal writer, schema, n_written
        if not rows:
            return
        if provider_columns:
            for r in rows:
                for k, v in provider_columns.items():
                    r.setdefault(k, v)
        table = pa.Table.from_pylist(rows)
        if writer is None:
            schema = table.schema
            writer = pq.ParquetWriter(str(output), schema)
        else:
            table = table.cast(schema)
        writer.write_table(table)
        n_written += len(rows)

    try:
        for rid, body, resp, captured_at, upstream_fp in pairs_iter:
            batch.append(_row(rid, body, resp, captured_at, upstream_fp))
            if len(batch) >= batch_size:
                _flush(batch)
                batch = []
        _flush(batch)
    finally:
        if writer is not None:
            writer.close()

    return n_written


def parse_bucket_uri(uri: str) -> tuple[str, str]:
    """Split ``hf://buckets/<owner>/<name>[/<path>]`` into
    ``("<owner>/<name>", "<path>")``."""
    if not uri.startswith(_BUCKET_PREFIX):
        raise ValueError(f"not a bucket URI: {uri!r}")
    rest = uri[len(_BUCKET_PREFIX) :]
    parts = rest.split("/", 2)
    if len(parts) < 2 or not parts[0] or not parts[1]:
        raise ValueError(
            f"bucket URI must be hf://buckets/<owner>/<name>[/<path>], "
            f"got {uri!r}"
        )
    bucket_id = f"{parts[0]}/{parts[1]}"
    path_in_bucket = parts[2] if len(parts) > 2 else ""
    return bucket_id, path_in_bucket


_FILENAME_SAFE = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."


def _slug(s: str) -> str:
    """Filename-safe slug. Strips ``org/`` prefix from HF model ids."""
    s = s.split("/")[-1]
    out = "".join(c if c in _FILENAME_SAFE else "-" for c in s)
    while "--" in out:
        out = out.replace("--", "-")
    return out.strip("-_.") or "x"


def _default_bucket_filename(
    agent: str | None = None,
    model: str | None = None,
    provider: str | None = None,
) -> str:
    """``train-[<agent>-<model>-<provider>-]<utc>-<hex>.parquet``."""
    import time
    import uuid

    ts = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    parts = ["train"]
    if agent:
        parts.append(_slug(agent))
    if model:
        parts.append(_slug(model))
    if provider:
        # Preserve hf-router/fireworks-ai → hf-router-fireworks-ai
        # (``_slug`` would otherwise strip everything before the last /).
        parts.append(_slug(provider.replace("/", "-")))
    parts.append(ts)
    parts.append(uuid.uuid4().hex[:6])
    return "-".join(parts) + ".parquet"


def push_bucket(
    capture_dir: Path | str,
    bucket_uri: str,
    *,
    model: str | None = None,
    agent: str | None = None,
    filename: str | None = None,
) -> int:
    """Export to parquet and upload to ``hf://buckets/<owner>/<name>[/<prefix>]``.
    Default filename is unique per call; pass ``filename`` to overwrite
    in place."""
    import tempfile

    from huggingface_hub import batch_bucket_files

    bucket_id, prefix = parse_bucket_uri(bucket_uri)
    prefix = prefix.rstrip("/")
    provider_columns = detect_provider_columns(capture_dir)
    if filename is None:
        filename = _default_bucket_filename(
            agent=agent,
            model=model,
            provider=provider_columns.get("provider") or None,
        )
    remote_path = f"{prefix}/{filename}" if prefix else filename

    with tempfile.TemporaryDirectory() as tmpdir:
        local_file = Path(tmpdir) / filename
        n_rows = export_local(
            capture_dir, local_file, provider_columns=provider_columns,
        )
        batch_bucket_files(bucket_id, add=[(str(local_file), remote_path)])

    return n_rows
