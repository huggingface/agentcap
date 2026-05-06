"""Trace dir → manifest export.

For each ``<request_id>.request.json`` in the trace dir, pair with the
matching ``<request_id>.response.json`` if present, render through the
manifest builder, and emit one parquet row per request.

Two destinations:

- Local file (``--output FILE.parquet``): single parquet on disk.
- Storage Bucket (``--push hf://buckets/owner/name/prefix/``): mutable
  object storage on the Hub. Append-by-prefix; each push lands a
  unique parquet file under the supplied prefix.

Dataset repos are deliberately not a destination here — they're
*replace-only* on push, which doesn't fit the trace-accumulation
workflow. Render to local with ``--output``, then ``hf upload`` if
you want a published Dataset repo.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from .manifest import build_manifest


_BUCKET_PREFIX = "hf://buckets/"


def detect_model(trace_dir: Path | str) -> str | None:
    """Return the unique ``body.model`` value across all captured
    requests in ``trace_dir``.

    Returns ``None`` if no captured request carries a ``model`` field
    (e.g. legacy traces). Raises ``ValueError`` if requests are for
    more than one distinct model — the dataset format never mixes
    models, so this is a hard error regardless of any ``--model``
    override at the CLI. Models with different ``@revision`` suffixes
    are considered the same model (the bare id is returned).
    """
    trace_dir = Path(trace_dir)
    seen: set[str] = set()
    for req_path in sorted(trace_dir.glob("*.request.json")):
        try:
            rec = json.loads(req_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        m = (rec.get("body") or {}).get("model")
        if isinstance(m, str) and m:
            seen.add(_bare_model_id(m))
    if len(seen) > 1:
        raise ValueError(
            f"trace dir contains requests for multiple models: "
            f"{sorted(seen)}. Datasets never mix models — split into "
            f"separate trace dirs and export each one independently."
        )
    return seen.pop() if seen else None


def _bare_model_id(model: str) -> str:
    """Strip ``@revision`` suffix so ``gemma-4-E4B-it`` and
    ``gemma-4-E4B-it@main`` are treated as the same id."""
    return model.split("@", 1)[0]


def _iter_pairs(
    trace_dir: Path,
) -> Iterator[tuple[str, dict, dict | None, int]]:
    """Yield (request_id, request_body, response_body, captured_at) for
    every captured request, in stable filename order."""
    for req_path in sorted(trace_dir.glob("*.request.json")):
        rec = json.loads(req_path.read_text())
        rid = rec.get("request_id") or req_path.stem.split(".")[0]
        captured_at = int(rec.get("captured_at", 0))
        body = rec.get("body") or {}
        resp_path = trace_dir / f"{rid}.response.json"
        resp_body: dict | None = None
        if resp_path.exists():
            resp_rec = json.loads(resp_path.read_text())
            if resp_rec.get("stream"):
                resp_body = {"stream": True, "raw": resp_rec.get("raw", "")}
            else:
                resp_body = resp_rec.get("body") or {}
        yield rid, body, resp_body, captured_at


def _row_for_arrow(manifest_row: dict) -> dict:
    """Serialise the polymorphic ``request``/``response`` bodies as JSON
    strings so Arrow's per-row schema inference doesn't choke on
    heterogeneous tool-schema fields. Consumers do
    ``json.loads(row["request"])`` to get the dict back."""
    return {
        "request_id": manifest_row["request_id"],
        "model": manifest_row["model"],
        "captured_at": manifest_row["captured_at"],
        "request": json.dumps(manifest_row["request"], ensure_ascii=False),
        "response": json.dumps(manifest_row["response"], ensure_ascii=False),
        "n_tokens": manifest_row["n_tokens"],
        "sections": manifest_row["sections"],
        "token_role": manifest_row["token_role"],
    }


def build_rows(trace_dir: Path | str, *, processor, model: str) -> list[dict]:
    """Render every captured request into a manifest row, in memory.
    Streaming consumers should call :func:`export_local` instead."""
    trace_dir = Path(trace_dir)
    rows: list[dict] = []
    for rid, body, resp, captured_at in _iter_pairs(trace_dir):
        rows.append(
            build_manifest(
                processor,
                model=model,
                request_id=rid,
                captured_at=captured_at,
                request_body=body,
                response_body=resp,
            )
        )
    return rows


def export_local(
    trace_dir: Path | str,
    output: Path | str,
    *,
    processor,
    model: str,
    batch_size: int = 32,
    progress: bool = True,
):
    """Render the trace dir into a single parquet file on disk.

    Streams via ``pyarrow.parquet.ParquetWriter`` in batches of
    ``batch_size`` so peak memory is bounded and a mid-render kill
    leaves a valid parquet up to the last flushed batch. Returns the
    number of rows written."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    trace_dir = Path(trace_dir)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    request_files = sorted(trace_dir.glob("*.request.json"))
    total = len(request_files)
    if total == 0:
        raise ValueError(f"no captured requests in {trace_dir}")

    if progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(
                _iter_pairs(trace_dir),
                total=total,
                desc=f"export {trace_dir.name}",
                unit="row",
            )
        except ImportError:
            iterator = _iter_pairs(trace_dir)
    else:
        iterator = _iter_pairs(trace_dir)

    writer: pq.ParquetWriter | None = None
    schema: pa.Schema | None = None
    batch: list[dict] = []
    n_written = 0

    def _flush(rows: list[dict]) -> None:
        nonlocal writer, schema, n_written
        if not rows:
            return
        table = pa.Table.from_pylist(rows)
        if writer is None:
            schema = table.schema
            writer = pq.ParquetWriter(str(output), schema)
        else:
            table = table.cast(schema)
        writer.write_table(table)
        n_written += len(rows)

    try:
        for rid, body, resp, captured_at in iterator:
            mrow = build_manifest(
                processor,
                model=model,
                request_id=rid,
                captured_at=captured_at,
                request_body=body,
                response_body=resp,
            )
            batch.append(_row_for_arrow(mrow))
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
    ``("<owner>/<name>", "<path>")``.

    The path component may be empty (push directly under the bucket
    root) or a prefix (a directory inside the bucket).
    """
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


def _default_bucket_filename() -> str:
    """Per-call unique ``train-<utc>-<hex>.parquet`` so successive
    pushes accumulate side-by-side under a prefix."""
    import time
    import uuid

    ts = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    return f"train-{ts}-{uuid.uuid4().hex[:6]}.parquet"


def push_bucket(
    trace_dir: Path | str,
    bucket_uri: str,
    *,
    processor,
    model: str,
    filename: str | None = None,
):
    """Render the trace dir into parquet and upload to a Storage Bucket.

    ``bucket_uri`` is ``hf://buckets/<owner>/<name>[/<prefix>]``. The
    parquet file lands at ``<prefix>/<filename>`` (or ``<filename>`` if
    no prefix). When ``filename`` is None the auto-generated default is
    unique per call, so pushes accumulate; pass an explicit name to
    overwrite in place."""
    import tempfile

    from huggingface_hub import batch_bucket_files

    bucket_id, prefix = parse_bucket_uri(bucket_uri)
    prefix = prefix.rstrip("/")
    if filename is None:
        filename = _default_bucket_filename()
    remote_path = f"{prefix}/{filename}" if prefix else filename

    with tempfile.TemporaryDirectory() as tmpdir:
        local_file = Path(tmpdir) / filename
        n_rows = export_local(
            trace_dir, local_file, processor=processor, model=model
        )
        batch_bucket_files(bucket_id, add=[(str(local_file), remote_path)])

    return n_rows


def load_processor(model: str):
    """Load an HF tokenizer/processor for ``model``.

    Deferred import so unit tests can drive the rendering paths with a
    fake processor without pulling in transformers.
    """
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model)
