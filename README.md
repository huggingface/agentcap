# agentcap

An end-to-end harness for running real coding agents at scale across
`(agent × model × corpus)` and publishing every interaction as a
[Hugging Face dataset](https://huggingface.co/docs/datasets). Drives
the agent through a corpus of prompts, captures every chat-completion
request and response byte-for-byte, and pushes the result to a
Hugging Face Dataset repo — so consumers can replay, render, or
analyse what the agent actually sent and got back, without
reconstructing it from a log.

The loop:

```
  corpus  ──►  sandboxed agent run  ──►  capture  ──►  export  ──►  publish  ──►  inspect
   ▲                                                                                │
   └────────────────── reuse for the next (agent, model) ──────────────────────────┘
```

Each stage is independently usable. The capture proxy can sit in
front of any agent you drive yourself. The export reads any
agentcap-compatible capture dir. The corpus is just a `tasks.txt`.

## What this repo provides

1. **Runner** (`agentcap run`) — drives a registered agent CLI
   (`hermes`, `opencode`, `goose`, `pi`) through the corpus.
   Multi-turn follow-ups via `--followup synthesized`; per-agent
   skill injection via `--skills`; per-run sandbox (bwrap on
   Linux, lima on macOS) so agents that write absolute paths
   can't leak into the host repo.
2. **Capture proxy** (internal to `agentcap run`) — a transparent
   OpenAI-compat HTTP proxy between the agent and the model server,
   dumping every request/response pair to disk. Backend-agnostic:
   works with `llama.cpp`, Inference Endpoints, Inference Providers,
   anything that speaks `/v1/chat/completions`. Capture is dumb —
   no tokenizer, no rendering, just persist the bytes.
3. **Dataset export** (`agentcap export`) — bundles the captured
   capture dir into a parquet, one row per chat-completion request,
   and pushes to a Hugging Face Dataset repo. Files land under
   `data/` so the Hub Dataset Viewer renders them automatically;
   consumers `load_dataset(...)`. The default filename embeds
   `(agent, model, provider)` so a single repo holds many tuples
   without aliasing.

## Architecture

```
  ┌─────────────────────────── runner ───────────────────────────────────┐
  │                                                                      │
  │  corpus ──► [agent CLI inside sandbox] ──HTTP──► [capture proxy] ──┐ │
  │       ▲             │                                  │            ▼│
  │       │             ▼                                  ▼       [model│
  │       │   final response text          <capture-dir>/*.{req,resp}.json server]
  │       │             │                                          ▲     │
  │       │             ▼                                          │     │
  │       └── [follow-up synthesizer] ─────── HTTP (bypasses proxy)┘     │
  │             (multi-turn, optional)                                   │
  └──────────────────────────────────────────────────────────────────────┘
                                       ▼
                               agentcap export
                                       │
                                       ▼
                       Hugging Face Dataset repo
                                       │
                                       ▼
                       Dataset Viewer / load_dataset
```

The synthesizer talks to the model server **directly**, around the
capture proxy, so the capture stays a clean record of agent ↔ model
interaction; the synthesizer's own LLM calls are an orchestration
detail and never land in the dataset.

The split is intentional. **Capture is dumb** — no tokenizer, no
chat-template render, no per-token labels — just persist the bytes.
**Export is a dumb data shuffle** — pair `.request.json` with
`.response.json`, serialise as parquet, stamp a couple of constant
provider columns. Token-level analysis is consumer-side: the raw
request body is preserved verbatim per row, so re-rendering through
the model's chat template is a 5-line job that doesn't need to live
in this repo.

## Why model + agent identity matters

Two different models running the same agent produce **different**
captures: the chat template is model-specific, so token boundaries,
special tokens, and tools-schema injection points all differ. The
agent's system prompt and tools also vary by build (Hermes, OpenCode,
Goose, … each emit their own).

Datasets are tagged with `model` and never mix models. The raw
`request` body is preserved verbatim per row so consumers can group,
filter, or recompute identifiers on whatever axis their analysis
needs.

## Sandbox prerequisites

Every `agentcap run` invocation executes the agent inside a per-agent
sandbox. Nothing on the host is visible to the agent except paths the
driver explicitly bind-mounts. The agent CLI itself lives **in the
image / VM, not on the host** — `agentcap run` builds it from a
declarative spec on first use, then reuses the build across runs.

### Linux

```bash
sudo apt install -y bubblewrap buildah        # one-time
# Ubuntu 24.04 only: allow unprivileged user namespaces for bwrap.
sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0
echo 'kernel.apparmor_restrict_unprivileged_userns=0' \
    | sudo tee /etc/sysctl.d/60-agentcap-bwrap.conf
```

Each agent has a Containerfile at `containers/agentcap-<agent>.Containerfile`
(`buildah bud` builds it). `agentcap run --agent <agent>` triggers
the build on first use — cold builds take 1–5 min, subsequent runs
reuse the image. A Containerfile change is detected via a label hash
and forces a rebuild.

### macOS

```bash
brew install lima                              # one-time
```

Each agent has a Lima template at `scripts/lima/agentcap-<agent>.yaml`.
`agentcap run --agent <agent>` provisions the per-agent VM on first
use (~30s cold boot) and reuses it on subsequent runs.

## Quick start

```bash
pip install -e .

# Pick a server. Three flavours, same proxy front-end.
#
#  (a) Inference Providers — zero infra, curated model catalogue,
#      pay per token. Right default for casual capture.
#      --upstream https://router.huggingface.co
#
#  (b) Inference Endpoints — dedicated GPU + specific model +
#      scale-to-zero. Right for repeatable runs against one model.
#
#  (c) Local llama.cpp — full control over quant and chat template.
#      Right for research that needs model-implementation detail.
GGUF_PATH=/path/to/model-Q4_K_M.gguf REASONING=off \
    ./scripts/start_llama_cpp_server.sh &

# Drive an agent through a corpus. Each run mints a fresh subdir
# under .agentcap/ in the workspace ($AGENTCAP_WORKSPACE or cwd):
#   .agentcap/<agent>-<provider>-<utc>/{captures,run.json,...}
agentcap run \
    --agent hermes \
    --model google/gemma-4-E4B-it \
    --upstream http://127.0.0.1:8000 \
    --tasks examples/transformers-coding-session/tasks.txt \
    --turns 4 --followup synthesized

# Run a second agent on the same corpus, side by side.
agentcap run \
    --agent goose \
    --model google/gemma-4-E4B-it \
    --upstream http://127.0.0.1:8000 \
    --tasks examples/transformers-coding-session/tasks.txt \
    --turns 4 --followup synthesized

# Browse what's captured.
agentcap ls

# Push everything to a Dataset repo.
agentcap export --all --push my-org/my-captures/<corpus>

# Or push selected runs only.
agentcap export hermes-local-20260512-162345 \
    --push my-org/my-captures/<corpus>

# Browse captured requests (fzf-driven picker with body preview;
# falls back to a plain table if fzf isn't on PATH).
agentcap inspect                       # everything in the workspace
agentcap inspect <run-id>              # one run only
agentcap inspect <request-id>          # dump a specific body

# Re-issue a single captured request to an OpenAI-compatible target.
agentcap replay <request-id> --target http://127.0.0.1:8000
# Compose with the picker:
agentcap replay $(agentcap inspect --rid) --target http://127.0.0.1:8000
```

See [docs/tested-models-and-agents.md](docs/tested-models-and-agents.md)
for which model + agent combinations have been validated end-to-end.

For `agentcap run`, `--model` is required for all drivers.

When `--followup synthesized` is enabled, `--synth-upstream` defaults
to `--upstream` and `--synth-model` defaults to `--model`. Pass synth
flags only when you intentionally want a different synth backend/model.

## Workspace and `agentcap ls`

`agentcap run` writes each invocation to
`$AGENTCAP_WORKSPACE/.agentcap/<agent>-<provider>-<utc>/` (or cwd if
`AGENTCAP_WORKSPACE` is unset). Pass `--workdir` to override.

```bash
agentcap ls                # short table
agentcap ls --long         # add upstream + per-run capture counts
```

## Pushing to a Dataset repo

`agentcap export` walks one or more runs and uploads each as a parquet
file into a Hugging Face Dataset repo (auto-created on first push).
Files land under `data/[<subdir>/]<file>.parquet` so the Hub Dataset
Viewer renders the parquet automatically.

```bash
# Push every run in the workspace.
agentcap export --all --push my-org/my-captures/<corpus>

# Or push selected runs (run-ids from `agentcap ls`).
agentcap export hermes-local-20260512-162345 goose-local-20260512-170000 \
    --push my-org/my-captures/<corpus>

# Or point at an arbitrary workdir / capture dir directly.
agentcap export ./some/workdir --push me/d
```

Each run lands as a unique parquet file. The default filename embeds
`(agent, model, provider)` so a single repo can hold many tuples
without aliasing —
`train-<agent>-<model>-<provider>-YYYYMMDDTHHMMSS-HEX6.parquet`.
`<agent>` is read from `run.json`; `<model>` and `<provider>` are
derived from the captured requests. Consumers read the union via
`load_dataset("my-org/my-captures")`.

One `agentcap export` invocation produces one git commit, regardless
of how many runs it bundles. On the first push to an empty repo,
agentcap also seeds a dataset card (`README.md`); subsequent pushes
leave any existing card untouched.

## What lands on disk

Per chat-completion request, two files in `<capture-dir>/`:

- `<request_id>.request.json` — `{request_id, captured_at,
  upstream_url, body}` where `body` is the raw OpenAI request.
  `upstream_url` is the proxy's configured upstream at capture
  time; the export layer uses it to derive `provider`.
- `<request_id>.response.json` — `{request_id, captured_at_resp,
  stream, status_code, body|raw, upstream_fingerprint}`. For
  streaming responses, `raw` holds the assembled SSE bytes
  verbatim; for non-streaming, `body` is the parsed JSON.
  `upstream_fingerprint` distils a few response headers (`Server`,
  `X-Served-By`, `Via`, `X-Build-Info`, body-echoed `model`) so
  per-row backend identity survives into the parquet.

No tokenisation, no rendering, no derived metadata — just the bytes.

## Parquet schema

`agentcap export` emits one row per captured request. Columns:

| column              | source                                  |
|---|---|
| `request_id`        | proxy-minted UUID                       |
| `model`             | `request.body.model`                    |
| `captured_at`       | request capture epoch                   |
| `request`           | JSON-stringified raw OpenAI request     |
| `response`          | JSON-stringified raw response (or `{stream: true, raw: "<SSE bytes>"}` for streamed) |
| `served_by`         | per-response `X-Served-By` header (HF Router sub-provider routing) |
| `served_build_info` | per-response `X-Build-Info` header      |
| `served_model`      | per-response body-echoed `model`        |
| `provider`          | derived from `upstream_url` hostname (constant per file) |
| `upstream_url`      | proxy upstream at capture time (constant per file) |

The `request` and `response` columns are JSON strings (not nested
structs) so Arrow doesn't infer a schema over heterogeneous tool-call
fields. Consumers `json.loads` them. To recover per-message token
ranges, render `request.messages` through the model's chat template
yourself — a 5-line job via `transformers.AutoTokenizer.apply_chat_template`.

## Server backends

The proxy is backend-agnostic. Document the menu so users pick the
right one per use case:

| backend | when to use |
|---|---|
| **Inference Providers** (`router.huggingface.co`) | demos, casual capture; zero infra; curated model catalogue; pay per token |
| **Inference Endpoints** | dedicated GPU + specific model + scale-to-zero between corpus runs; OpenAI-compat by default |
| **Local `llama.cpp` server** (`llama serve`) | full control over quant / chat template / sampler; required for research that depends on model-implementation detail (e.g. kv-cache-reuse splice work) |
| `transformers serve` | works for small models, awkward for big ones at long context |

For which (backend, model, agent) combinations have been validated
end-to-end, see [docs/tested-models-and-agents.md](docs/tested-models-and-agents.md).

## Roadmap

See [ROADMAP.md](ROADMAP.md).

## Running tests

```bash
pip install -e '.[dev]'
pytest tests/
```

Live driver tests in [tests/test_drivers_live.py](tests/test_drivers_live.py)
run when a model endpoint is reachable, skip otherwise. Either set
`AGENTCAP_TEST_LLM_URL=http://host:port/v1`, or have the `llama`
executable on `$PATH` so the fixture spawns one via `llama serve`
(install with `curl -fsSL https://llama.app/install.sh | sh`).
Override the agent's model id with `AGENTCAP_TEST_MODEL` (default
`gemma-4-E4B-it`).

The per-agent sandbox is built / booted lazily on first use (same
lifecycle as `agentcap run`), so the first session pays a multi-
minute cold-build per agent.

## License

Apache 2.0 — see [LICENSE](LICENSE).
