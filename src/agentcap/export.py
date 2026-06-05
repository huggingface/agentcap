"""Capture dir → parquet export.

For each ``<request_id>.request.json``, pair with the matching
``<request_id>.response.json`` and emit one parquet row.

Destination: ``--push <owner>/<name>[/<subdir>]`` — uploaded into a
Hugging Face Dataset repo. Files under ``data/`` get the Hub Dataset
Viewer automatically.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from .provider import _hostname_fallback, refine_for_sub_provider


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
) -> Iterator[tuple[str, dict, dict | None, int, dict, str | None, int | None]]:
    """Yield (request_id, request_body, response_body, captured_at,
    upstream_fingerprint, task_id, turn) per captured request, in
    filename order. ``task_id`` / ``turn`` come from the wrapping
    ``.request.json`` record (orchestrator-side metadata that isn't
    inside the OpenAI body) — preserving them in the parquet lets
    downstream picker UIs group + index rows without having to fall
    back to ``-``."""
    for req_path in sorted(capture_dir.glob("*.request.json")):
        rec = json.loads(req_path.read_text())
        rid = rec.get("request_id") or req_path.stem.split(".")[0]
        captured_at = int(rec.get("captured_at", 0))
        body = rec.get("body") or {}
        task_id = rec.get("task_id")
        turn = rec.get("turn")
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
        yield rid, body, resp_body, captured_at, upstream_fp, task_id, turn


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
    task_id: str | None = None,
    turn: int | None = None,
) -> dict:
    # request / response stringified so Arrow doesn't infer a schema over
    # heterogeneous tool-schema fields. Consumers json.loads them.
    model = (request_body.get("model") or "") if isinstance(request_body, dict) else ""
    return {
        "request_id": request_id,
        "model": model,
        "captured_at": captured_at,
        "task_id": task_id,
        "turn": turn,
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
        for rid, body, resp, captured_at, upstream_fp, task_id, turn in pairs_iter:
            batch.append(
                _row(rid, body, resp, captured_at, upstream_fp, task_id, turn)
            )
            if len(batch) >= batch_size:
                _flush(batch)
                batch = []
        _flush(batch)
    finally:
        if writer is not None:
            writer.close()

    return n_written


def parse_collection_base(uri: str) -> tuple[str, str]:
    """Split ``<owner>/<base>`` (optionally prefixed with
    ``hf://datasets/``) into ``("<owner>", "<base>")``.

    ``<base>`` drives all three artifacts: captures dataset
    ``<owner>/<base>-captures``, traces dataset ``<owner>/<base>-traces``,
    and the HF Collection of the same title under ``<owner>``."""
    s = uri.removeprefix("hf://datasets/").strip("/")
    parts = s.split("/")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(
            f"--push must be <owner>/<base>, got {uri!r}"
        )
    return parts[0], parts[1]


def captures_repo_id(owner: str, base: str) -> str:
    return f"{owner}/{base}-captures"


_FILENAME_SAFE = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."


def _slug(s: str) -> str:
    """Filename-safe slug. Strips ``org/`` prefix from HF model ids."""
    s = s.split("/")[-1]
    out = "".join(c if c in _FILENAME_SAFE else "-" for c in s)
    while "--" in out:
        out = out.replace("--", "-")
    return out.strip("-_.") or "x"


def _default_filename(
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


_CAPTURES_README_TEMPLATE = """\
---
license: apache-2.0
tags:
- agentcap
- agentcap-captures
---

# {repo_id}

HTTP captures of agent ↔ model interactions — one parquet row per
`/v1/chat/completions` call. Produced by
[agentcap](https://github.com/huggingface/agentcap).

Native session traces for the same runs live in companion datasets
named `{base}-<agent>-traces`. They're all grouped under the
[{collection_title} Collection](https://huggingface.co/{owner})
alongside this dataset. Join on `run_id`.

## Loading

```python
from datasets import load_dataset

ds = load_dataset("{repo_id}", split="train")
```

## Schema

| column | description |
|---|---|
| `run_id` | agentcap run id; matches the per-run folder in the traces dataset |
| `request_id` | UUID minted by the capture proxy |
| `model` | Model id from the captured request body |
| `captured_at` | Epoch seconds when the request was captured |
| `request` | Raw OpenAI request body, JSON-stringified |
| `response` | Raw OpenAI response body, JSON-stringified (or `{{"stream": true, "raw": ...}}` for SSE) |
| `served_by` | Per-response `X-Served-By` header (HF Router sub-provider routing) |
| `served_build_info` | Per-response `X-Build-Info` header |
| `served_model` | Per-response body-echoed `model` |
| `provider` | Derived from the proxy upstream URL (constant per file) |
| `upstream_url` | Proxy upstream URL at capture time (constant per file) |

`request` and `response` are JSON strings; consumers `json.loads(...)`
them. To recover per-message token ranges, render `request.messages`
through the model's chat template yourself —
`transformers.AutoTokenizer.apply_chat_template`.
"""


_TRACES_README_TEMPLATE = """\
---
license: apache-2.0
tags:
- agent-traces
- agentcap
- agentcap-traces
- agentcap-traces-{agent}
source_datasets:
- {captures_repo}
---

# {repo_id}

{agent} coding-agent session traces produced by
[agentcap](https://github.com/huggingface/agentcap) runs. Each run
contributes one folder under `data/<run_id>/`; inside, one file per
session in `{agent}`'s native export format.

The on-the-wire HTTP captures for these same runs live in
[{captures_repo}](https://huggingface.co/datasets/{captures_repo}).
Both belong to the
[{collection_title} Collection](https://huggingface.co/{owner})
— join on `run_id` to align captures with traces.
"""


def traces_repo_id_for(owner: str, base: str, agent: str) -> str:
    """Per-agent traces dataset id. One agent per dataset keeps the
    schema homogeneous — the Hub viewer can't reconcile pi's
    type-discriminated events with goose's session-as-object dump."""
    return f"{owner}/{base}-{agent}-traces"


def _captures_readme(
    *,
    repo_id: str,
    owner: str,
    base: str,
    collection_title: str,
) -> str:
    return _CAPTURES_README_TEMPLATE.format(
        repo_id=repo_id,
        owner=owner,
        base=base,
        collection_title=collection_title,
    )


def _traces_readme(
    *,
    repo_id: str,
    captures_repo: str,
    owner: str,
    collection_title: str,
    agent: str,
) -> str:
    return _TRACES_README_TEMPLATE.format(
        repo_id=repo_id,
        captures_repo=captures_repo,
        owner=owner,
        collection_title=collection_title,
        agent=agent,
    )


def push_captures_dataset(
    items: list[dict],
    *,
    owner: str,
    base: str,
) -> tuple[str, list[int]]:
    """Render N capture dirs to parquet under ``<owner>/<base>-captures``
    in a single commit. Returns ``(repo_id, [n_rows...])``.

    ``items`` is a list of dicts, each with:
      - ``capture_dir`` (required): path to a capture dir
      - ``model`` (required): model id used in the default filename
      - ``agent`` (optional): agent name embedded in the default filename
      - ``run_id`` (optional): stamped onto every row + into the filename
      - ``filename`` (optional): overrides the default unique name

    The repo is created on first push (``exist_ok=True``); files land
    under ``data/<filename>.parquet`` so the Hub Dataset Viewer picks
    them up automatically.
    """
    import tempfile

    from huggingface_hub import CommitOperationAdd, HfApi

    repo_id = captures_repo_id(owner, base)
    api = HfApi()
    api.create_repo(
        repo_id=repo_id, repo_type="dataset",
        private=True, exist_ok=True,
    )

    # Seed a dataset card on first push (no README in the repo yet).
    # Later pushes leave any existing README alone — including
    # user-edited ones.
    try:
        existing = set(api.list_repo_files(repo_id, repo_type="dataset"))
    except Exception:
        existing = set()
    include_readme = "README.md" not in existing

    n_rows_list: list[int] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        operations: list[CommitOperationAdd] = []
        if include_readme:
            operations.append(CommitOperationAdd(
                path_in_repo="README.md",
                path_or_fileobj=_captures_readme(
                    repo_id=repo_id,
                    owner=owner,
                    base=base,
                    collection_title=base,
                ).encode("utf-8"),
            ))
        for i, item in enumerate(items):
            cap_dir = item["capture_dir"]
            model = item["model"]
            agent = item.get("agent")
            run_id = item.get("run_id")
            filename = item.get("filename")
            provider_columns = detect_provider_columns(cap_dir)
            extra_columns = dict(provider_columns)
            if run_id:
                extra_columns["run_id"] = run_id
            if filename is None:
                filename = _default_filename(
                    agent=agent,
                    model=model,
                    provider=provider_columns.get("provider") or None,
                )
            path_in_repo = f"data/{filename}"
            local_file = Path(tmpdir) / f"{i}-{filename}"
            n_rows = export_local(
                cap_dir, local_file, provider_columns=extra_columns,
                progress=False,
            )
            n_rows_list.append(n_rows)
            operations.append(CommitOperationAdd(
                path_in_repo=path_in_repo,
                path_or_fileobj=str(local_file),
            ))

        api.create_commit(
            repo_id=repo_id,
            repo_type="dataset",
            operations=operations,
            commit_message=f"agentcap export: add {len(operations)} parquet(s)",
        )

    return repo_id, n_rows_list


def push_agent_traces_dataset(
    items: list[dict],
    *,
    owner: str,
    base: str,
    agent: str,
) -> tuple[str, int]:
    """Upload raw trace files for ONE agent under
    ``<owner>/<base>-<agent>-traces`` in a single commit. Returns
    ``(repo_id, n_files_total)``.

    ``items`` is a list of dicts, each with:
      - ``traces_dir`` (required): path to a ``<run>/traces/`` dir
      - ``run_id`` (required): folder name in the dataset repo

    Splitting by agent (one dataset per agent) keeps each dataset's
    schema homogeneous — the Hub viewer can't reconcile pi's
    type-discriminated events with goose's session-as-object dump.

    Files are uploaded **as-is** — no JSON parsing, no schema
    transformation. Empty trace dirs contribute 0 files. Returns 0
    files when the entire item list has no files; the repo is still
    created so the collection link stays consistent.
    """
    from huggingface_hub import CommitOperationAdd, HfApi

    repo_id = traces_repo_id_for(owner, base, agent)
    captures_repo = captures_repo_id(owner, base)
    api = HfApi()
    api.create_repo(
        repo_id=repo_id, repo_type="dataset",
        private=True, exist_ok=True,
    )

    try:
        existing = set(api.list_repo_files(repo_id, repo_type="dataset"))
    except Exception:
        existing = set()
    include_readme = "README.md" not in existing

    operations: list[CommitOperationAdd] = []
    if include_readme:
        operations.append(CommitOperationAdd(
            path_in_repo="README.md",
            path_or_fileobj=_traces_readme(
                repo_id=repo_id,
                captures_repo=captures_repo,
                owner=owner,
                collection_title=base,
                agent=agent,
            ).encode("utf-8"),
        ))

    n_files = 0
    for item in items:
        traces_dir = Path(item["traces_dir"])
        run_id = item["run_id"]
        if not traces_dir.is_dir():
            continue
        for f in sorted(p for p in traces_dir.iterdir() if p.is_file()):
            operations.append(CommitOperationAdd(
                path_in_repo=f"data/{run_id}/{f.name}",
                path_or_fileobj=str(f),
            ))
            n_files += 1

    # Only commit if we have something to add. If even the README is
    # already up, skip the empty commit silently.
    if not operations:
        return repo_id, n_files
    api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        operations=operations,
        commit_message=(
            f"agentcap export: add {agent} traces "
            f"({n_files} file(s) across {len(items)} run(s))"
        ),
    )

    return repo_id, n_files


def ensure_collection(
    *,
    owner: str,
    base: str,
    repos: list[str],
) -> str:
    """Find-or-create the ``<owner>/<base>`` collection and ensure every
    repo in ``repos`` is an item. Returns the collection slug.

    Idempotent: existing items are kept (``exists_ok=True``)."""
    from huggingface_hub import HfApi

    api = HfApi()
    slug: str | None = None
    try:
        for c in api.list_collections(owner=owner, q=base, limit=20):
            if c.title == base:
                slug = c.slug
                break
    except Exception:
        slug = None

    if slug is None:
        col = api.create_collection(
            title=base,
            namespace=owner,
            description=(
                "agentcap: paired HTTP captures + native session "
                "traces. Join on run_id."
            ),
            private=True,
            exists_ok=True,
        )
        slug = col.slug

    for repo in repos:
        try:
            api.add_collection_item(
                collection_slug=slug,
                item_id=repo,
                item_type="dataset",
                exists_ok=True,
            )
        except Exception:
            # Item-add isn't load-bearing — the README cross-links
            # already make the relationship discoverable. Keep going.
            pass

    return slug
