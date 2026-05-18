# Plan: proxy-owned trace metadata + upstream drift detection

## Target layout

```
<trace_dir>/
  _proxy.json          ← proxy writes ONCE at startup
  <rid>.request.json   ← per request
  <rid>.response.json  ← per response (now includes per-call fingerprint)
```

`agentcap run` keeps writing `<workdir>/run.json` (its own summary) but
**outside** `trace_dir`, and export never reads it.

## Step 1 — Proxy owns its startup probe

- `serve()` / `serve_in_thread()` accept an `api_key` argument.
- At startup, call the existing `probe(upstream, api_key=...)`.
- Write the full probe to `<trace_dir>/_proxy.json`:
  - `upstream_url`, `provider`, `probed_at`, `endpoints` (full payload —
    includes `build_info` etc.; whatever the probe surfaced)
  - `startup_fingerprint`: small struct distilled from probe + first
    probe HTTP response headers (see Step 3 for shape)
- Single write, before the first agent request can possibly land.

## Step 2 — `agentcap run` stops writing `_meta.json`

- Remove the `_meta.json` write block from `run_cmd` (it currently
  shadows the proxy's territory).
- Probe call moves into the proxy's `serve()`.
- `run.json` still gets written at `<workdir>/run.json` for orchestrator
  concerns (`agent`, `model`, completed turns, …); export never reads it.

## Step 3 — Per-response upstream fingerprint

On every forwarded upstream response, the proxy extracts a small
fingerprint and stamps it onto the captured `.response.json`:

```json
"upstream_fingerprint": {
  "server": "llama.cpp",            // Server: response header
  "x_served_by": null,              // HF Router routing hint when present
  "via": null,                      // CDN / reverse-proxy chain (rare)
  "build_info": "b9039-4f04476e5",  // when present (llama.cpp echoes it on some responses; otherwise mirror startup)
  "served_model": "qwen3.6-35b-a3b" // body.model echoed by the server (sometimes differs from request.model on routers)
}
```

Cheap — no extra HTTP round trips, header parsing only.

## Step 4 — Drift detection

- Proxy keeps the `startup_fingerprint` in memory after the probe.
- On every response, compare to current fingerprint. On any field
  changing:
  - Append to in-memory
    `drift_events: [{request_id, timestamp, before, after}]`.
  - Log a single stderr warning the first time per-field (don't spam).
- On proxy shutdown, merge `drift_events` into `_proxy.json` (atomic
  rewrite via temp + rename).
- The HF Router sub-provider case is naturally handled: if
  `Llama-3.3-70B:fireworks-ai` lands on Fireworks for some requests and
  Together for others, `x_served_by` will differ and drift events
  surface it row-by-row.

## Step 5 — Export reads from `_proxy.json`

- `detect_provider_columns` reads `_proxy.json` instead of `_meta.json`.
- New parquet columns (per row, from `.response.json`'s
  `upstream_fingerprint`): `served_by`, `served_build_info`,
  `served_model`. The startup-derived `provider` / `upstream_url` /
  `server_version` stay flat (constant per file).

## Step 6 — Standalone proxy parity

- `agentcap proxy --upstream X --trace-dir Y` produces an export-ready
  trace dir with no orchestrator involvement. The "agent run is just an
  agent run" rule holds.

## Step 7 — Tests

- Existing `test_provider.py` keeps working (pure-Python classifier
  didn't move).
- New:
  - Proxy writes `_proxy.json` at startup with synthesized probe.
  - Per-response fingerprint extraction with httpx response mocks.
  - Drift event fires when `Server` header changes mid-run, doesn't
    fire when stable.
  - Export reads `_proxy.json` (covers both presence + absence/legacy
    paths).
- Drop/rename existing tests that asserted on `_meta.json` placement.

## Step 8 — Compatibility / migration

- Old bucket parquets keep working — they have neither file, just the
  columns we already added.
- `scripts/bucket_update.sh` switches to writing `_proxy.json`
  (synthesized for historical captures) instead of `_meta.json`. One
  renaming hop.
- For trace dirs still on disk with `_meta.json` (the in-progress
  qwen3.6 hermes run), export reads either filename during a transition
  window, prefers `_proxy.json`.

## Open questions (call before implementing)

- **Filename**: `_proxy.json` vs `proxy.json` (no underscore = looks
  like a capture file). I'd pick `_proxy.json` to match the underscore
  convention for "infra" files, but it's not a hill.
- **Atomic-rewrite on drift**: writing on every drift event vs. only at
  shutdown. Shutdown is simpler; mid-run-write is more robust to
  crashes. Start with shutdown, escalate if it bites.
- **HF Router header surface**: I don't know off the top of my head
  which headers HF Router actually sets — `x-served-by`? `cf-ray`?
  Will need a real probe to confirm which fields are worth pulling.

## Design discussion notes

Facts surfaced while reviewing how the three components share
information across the two supported use cases.

### Supported use cases

- **Automated agent run** — orchestrator (`agentcap run`) drives an
  agent CLI through a corpus against a proxy in front of a local or
  remote server.
- **Interactive session** — user points their own client at a
  standalone proxy (`agentcap proxy`) in front of a local or remote
  server.

Local vs. remote upstream is not a meaningful behavioural
distinction for the components — same probe paths, same fallbacks.

### Proxy's only inputs

Listen address, upstream URL, storage location. Nothing else.

- **No `--api-key` flag.** The agent (automated mode) or client
  (interactive mode) already puts the key on every request's
  `Authorization` header. The proxy never needs it as separate
  config — it can snag it off the wire when it wants to probe.
- **No `--agent` flag.** A one-time-at-startup flag would bake in
  an assumption that breaks the moment someone switches agents
  against a long-running proxy without restarting. Agent identity
  is supplied at export time instead, where the trace dir is the
  unit and per-row provenance can be reconstructed from captured
  request headers if the consumer needs that granularity.

### Working principle

Each component extracts as much as it can from its own observation
post. Cross-component metadata files are opportunistic enrichment,
never load-bearing. User input is the last resort, only when
nothing in the system can answer.

### Information needs at export time

- **`model`** — always derivable from `body.model` in every captured
  request. Both use cases, no help needed.
- **`provider type`** — free from upstream URL hostname; enriched
  by the startup probe when available; per-row `x-served-by`-style
  fingerprint surfaces sub-provider routing.
- **`agent identity`** — supplied at export time via
  `agentcap export --agent <name>`. Single rule for both modes:
  in the automated case the orchestrator's export-step wrapper
  passes it through; in the interactive case the user supplies it.
  Absent → column stays empty. Header sniffing (once request
  headers are captured) is an optional best-effort fallback.

These three columns are non-negotiable export outputs.

### Upstream drift detection

Pin a fingerprint of the upstream at startup (`Server`,
`x-served-by`, `build_info`, body-echoed `model`), compare every
response. Genuinely useful for HF Router sub-provider routing
where the same model id resolves to different backends row-to-row.
Free byproduct of per-response header capture.

### Agent drift detection — open

Mirror of upstream drift but on the request side: pin the first
request's identifying headers, compare each subsequent request,
log on mismatch. Useful if multiple agents accidentally share a
proxy in the interactive case, and as a self-check in the
automated case.

Open question: which headers actually identify each agent? Our
expectations, to be verified against real captures:

- **hermes** — uses openai-python; `User-Agent: OpenAI/Python <ver>`
  + `X-Stainless-*`. SDK-level, not agent-distinguishing on its own.
- **goose** — Rust; likely `User-Agent: reqwest/<ver>` unless
  overridden. Needs checking.
- **opencode** — TS/Node; potentially `User-Agent: opencode/<ver>`.
  Needs checking.
- **pi** — unknown.

Prerequisite: the proxy currently captures only request bodies;
to enable agent drift detection it also needs to persist request
headers on each `.request.json`. Once captured, real traces from
each supported agent answer the "which header?" question
empirically.
