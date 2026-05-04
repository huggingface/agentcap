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


def build_rows(trace_dir: Path | str, *, processor, model: str) -> list[dict]:
    """Render every captured request into a manifest row, in memory."""
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
):
    """Render the trace dir into a single parquet file on disk."""
    from datasets import Dataset

    rows = build_rows(trace_dir, processor=processor, model=model)
    ds = Dataset.from_list(rows)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(str(output))
    return ds


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
    """Per-call unique parquet filename for bucket pushes.

    Sortable by time (UTC), with a short random suffix so two pushes
    in the same second don't collide. Follows the ``train-*.parquet``
    convention so a future ``load_dataset`` over the prefix glob picks
    up every shard naturally.
    """
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
    parquet file lands at ``<prefix>/<filename>`` (or ``<filename>``
    if no prefix).

    By default ``filename`` is auto-generated per call, so successive
    pushes to the same prefix accumulate side-by-side instead of
    overwriting. Pass an explicit ``filename`` to opt back into
    overwrite-in-place (e.g. a "latest" pointer file).
    """
    import tempfile

    from datasets import Dataset
    from huggingface_hub import batch_bucket_files

    bucket_id, prefix = parse_bucket_uri(bucket_uri)
    prefix = prefix.rstrip("/")
    if filename is None:
        filename = _default_bucket_filename()
    remote_path = f"{prefix}/{filename}" if prefix else filename

    rows = build_rows(trace_dir, processor=processor, model=model)
    ds = Dataset.from_list(rows)

    with tempfile.TemporaryDirectory() as tmpdir:
        local_file = Path(tmpdir) / filename
        ds.to_parquet(str(local_file))
        batch_bucket_files(bucket_id, add=[(str(local_file), remote_path)])

    return ds


def load_processor(model: str):
    """Load an HF tokenizer/processor for ``model``.

    Deferred import so unit tests can drive the rendering paths with a
    fake processor without pulling in transformers.
    """
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model)
