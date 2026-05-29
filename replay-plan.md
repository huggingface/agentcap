# Plan: `agentcap replay` — re-issue a captured request

## Why this exists

agentcap stores captured requests in a format consumers shouldn't
need to learn. Today anyone wanting to re-send a captured request
to an OpenAI-compatible endpoint re-implements agentcap's read path
(kv-reuse-lab does this in three places: `_fetch_pair_bodies`,
`fetch_bodies_by_request_id`, plus a manual parquet schema). Replay
is the natural agentcap-owned primitive.

**Single-turn only.** Multi-turn replay diverges the moment the new
model responds differently, so conversation replay stays out of
scope. What ships is: *resolve a captured request by id, hand back
the body, optionally POST it.*

## Honor the hardened decisions

AGENTS.md #1–3 still hold. Replay must be:

- **Dumb transport.** No tokenizer, no chat-template render, no
  schema invention. Bytes in, bytes out.
- **Byte-faithful.** The captured body is sent verbatim. No
  normalisation, no flags that mutate it. The repo's example
  corpora contain zero null-valued JSON paths; consumers that hit
  null-rejection against strict upstreams (e.g. their own corpus
  against a stricter llama-server build) own their own
  normalisation step — same as today.

## Target surface

Two layers, mirroring `export.py` / `export_cmd`. The library API
is the real reuse surface; the CLI is convenience.

### Library API — `src/agentcap/replay.py`

```python
def load_request(source: str, request_id: str) -> dict:
    """Return the raw captured request body for ``request_id``.

    ``source`` resolves any of:
      - a local capture dir (``<rid>.request.json`` files),
      - a local ``.parquet`` exported by ``agentcap export``,
      - ``hf://datasets/<owner>/<name>[/<subdir>]`` (or the bare
        ``<owner>/<name>`` form) resolved via ``HfFileSystem``.

    Raises ``KeyError`` if the id is not found.
    """

def load_requests(source: str, request_ids: Iterable[str]) -> dict[str, dict]:
    """Batch form: one pass over the source, returns ``{id: body}``.
    Replaces the lab's ``_fetch_pair_bodies`` /
    ``fetch_bodies_by_request_id``."""
```

That's the whole surface. Reuse `export.parse_dataset_uri` and the
`HfFileSystem` + pyarrow pattern already in `export.py`.

No `send()`, no `extra_params`, no `strip_nulls`. Sending is one
`httpx.post` line the caller already has; param merging is
`body | extra_params`; null-stripping is a caller-side
normalisation. Keeping these out of the library preserves AGENTS.md
#3 and keeps consumers in control of their own HTTP, retries,
timeouts, and normalisation policy.

### CLI — `agentcap replay`

```
agentcap replay <request-id> --target <url> [--source <dir|parquet|hf-uri>]
```

- `<request-id>` resolves against the workspace by default — the
  same way `export` and `ls` do. So `agentcap replay <rid> --target
  http://127.0.0.1:8080` works straight after a local
  `agentcap run`, no `--source` needed.
- POSTs the captured body verbatim to
  `<target>/v1/chat/completions`, prints the response to stdout,
  logs status / timing to stderr.
- No flags beyond `--source` and `--target`. Byte-faithful, no
  knobs.

## Steps

1. **`src/agentcap/replay.py`**: implement `load_request` and
   `load_requests`. Factor the local-dir read out of
   `export._iter_pairs` if cleanly shareable; otherwise a small
   local reader (dir reads `body` from `<rid>.request.json`;
   parquet/hf reads + `json.loads` the `request` column). Reuse
   `parse_dataset_uri`.
2. **`__main__.py`**: add `replay_cmd`; lift the workspace
   resolution shape from `export_cmd`.
3. **Tests** (`tests/test_replay.py`): (a) `load_request` round-trips
   a body from a local capture-dir fixture and from a small local
   parquet written by `export_local`; (b) the CLI posts to a local
   Starlette echo app (pattern from `tests/test_proxy_http.py`) and
   prints the response verbatim.
4. **Docs**: move Replay in `ROADMAP.md` from "Out of scope" to a
   shipped feature with the scoped definition above (keep the
   *conversation*-replay-is-meaningless note). Add an `AGENTS.md`
   decision: "Replay is byte-faithful: no normalisation, no flags
   that mutate the body." Add a `replay` row to the README command
   list.

## What kv-reuse-lab does after this

Deletes `_fetch_pair_bodies` and `fetch_bodies_by_request_id`;
replaces them with
`agentcap.replay.load_requests(source, {donor_id, recipient_id})`.
Keeps `_strip_nulls`, its httpx client, slot topology, donor→
recipient ordering, stderr/metrics parsing — all of which are
genuinely the lab's job. The coupling to agentcap's storage layout
is gone; the only remaining shared contract is the OpenAI body
shape, which is irreducible.
