# agentcap

An end-to-end harness for running real coding agents at scale across
`(agent × model × corpus)` and publishing every interaction as a
reproducible [Hugging Face dataset](https://huggingface.co/docs/datasets).
Drives the agent through a corpus of prompts, captures every
chat-completion byte the agent sends, renders it through the model's
chat template with per-message token boundaries, and pushes the
result to a Storage Bucket — without ever reconstructing what the
agent did from a log.

The loop:

```
  corpus  ──►  sandboxed agent run  ──►  capture  ──►  export  ──►  publish  ──►  inspect
   ▲                                                                                │
   └────────────────── reuse for the next (agent, model) ──────────────────────────┘
```

Each stage is independently usable. The capture proxy can sit in
front of any agent you drive yourself. The export reads any
agentcap-compatible trace dir. The corpus is just a `tasks.txt`.

## What this repo provides

1. **Corpus authoring** (`agentcap corpus`) — generate a `tasks.txt`
   from real workloads: GitHub issues, PR descriptions, forum
   threads, or an LLM meta-prompt. Hand-written corpora carry
   author blind spots; real-issue corpora exercise the long tail
   of how users phrase things.
2. **Runner** (`agentcap run`) — drives a registered agent CLI
   (Hermes, OpenCode, Goose, pi-mono today) through the corpus.
   Multi-turn follow-ups via `--followup synthesized`; per-agent
   skill injection via `--skills`; per-run sandbox (bwrap on
   Linux, lima on macOS) so agents that write absolute paths
   can't leak into the host repo.
3. **Capture proxy** (`agentcap proxy`) — a transparent OpenAI-compat
   HTTP proxy between the agent and the model server, dumping every
   request/response pair to disk. Backend-agnostic: works with
   `llama.cpp`, Inference Endpoints, Inference Providers, anything
   that speaks `/v1/chat/completions`. Capture is intentionally
   dumb — no tokenizer, no rendering, just persist the bytes.
4. **Dataset export** (`agentcap export`) — renders the captured
   trace dir into a parquet with chat-template-aware per-message
   token ranges and role labels. Push directly to a Hugging Face
   Storage Bucket, append-by-prefix, Xet-deduplicated. Filename
   embeds `(agent, model)` so a single prefix holds many tuples
   without aliasing.

A consumer side, separately:

5. **Inspector** (planned, hosted Space) — pure parquet reader
   that surfaces a session timeline (chat-style with expandable
   tool calls), the rendered byte stream the model actually
   received (the differentiator), and cross-(agent, model)
   comparisons on the same task.

## Architecture

```
  ┌─────────────────────────── runner ───────────────────────────────────┐
  │                                                                      │
  │  corpus ──► [agent CLI inside sandbox] ──HTTP──► [capture proxy] ──┐ │
  │       ▲             │                                  │            ▼│
  │       │             ▼                                  ▼       [model│
  │       │   final response text          <trace-dir>/*.{req,resp}.json server]
  │       │             │                                          ▲     │
  │       │             ▼                                          │     │
  │       └── [follow-up synthesizer] ─────── HTTP (bypasses proxy)┘     │
  │             (multi-turn, optional)                                   │
  └──────────────────────────────────────────────────────────────────────┘
                                       ▼
                               agentcap export
                                       │
                                       ▼
                       Hugging Face Storage Bucket
                                       │
                                       ▼
                            inspector Space / load_dataset
```

The synthesizer talks to the model server **directly**, around the
capture proxy, so the trace stays a clean record of agent ↔ model
interaction; the synthesizer's own LLM calls are an orchestration
detail and never land in the dataset.

The split is intentional. **Capture is dumb** — no tokenizer, no
chat-template render, no per-token labels — just persist the bytes.
**Export is smart** — loads the model's tokenizer, renders, computes
per-message token ranges and role labels.

## Why model + agent identity matters

Two different models running the same agent produce **different**
traces: the chat template is model-specific, so token boundaries,
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
#      --upstream https://router.huggingface.co/v1
#
#  (b) Inference Endpoints — dedicated GPU + specific model +
#      scale-to-zero. Right for repeatable runs against one model.
#
#  (c) Local llama.cpp — full control over quant and chat template.
#      Right for research that needs model-implementation detail.
GGUF_PATH=/path/to/model-Q4_K_M.gguf REASONING=off \
    ./scripts/start_llama_cpp_server.sh &

# Drive Hermes through 30 prompts × 4 turns, capture, then export.
# --followup synthesized hits a small synth model for topic-aware
# multi-turn follow-ups (the synth call bypasses the capture proxy).
# --followup continue is the cheaper alternative (literal "continue").
agentcap run \
    --agent hermes \
    --upstream http://127.0.0.1:8000 \
    --tasks examples/transformers-coding-session/tasks.txt \
    --turns 4 --followup synthesized \
    --workdir runs/run-001/

# --workers parallelises the per-row chat-template render. On a
# 1000-row trace dir this is the difference between minutes and an
# hour.
agentcap export runs/run-001/traces \
    --output runs/run-001.parquet --workers 8
```

See [docs/tested-models-and-agents.md](docs/tested-models-and-agents.md)
for which model + agent combinations have been validated end-to-end.

`--model` is inferred from the captured request bodies; pass it
explicitly to override or when traces lack a model field. The trace
dir must be for a single model (the dataset format never mixes them).

`agentcap run` starts the capture proxy in-process and configures the
agent to talk through it. If you'd rather drive the agent yourself,
run `agentcap proxy` standalone and adjust your agent configuration
to point to the proxy.

## Pushing to a Storage Bucket

`--push` writes the parquet directly into a [Hugging Face Storage
Bucket](https://huggingface.co/docs/hub/storage-buckets) — mutable,
append-by-prefix, Xet-deduplicated:

```bash
agentcap export <trace-dir> --push hf://buckets/my-org/my-traces/<corpus>/
```

Each run lands as a unique parquet file under the supplied prefix.
The default filename embeds the agent and model so a single bucket
prefix can hold many `(agent, model)` tuples without aliasing —
`train-<agent>-<model>-YYYYMMDDTHHMMSS-HEX6.parquet`. Agent is
auto-detected from `<trace-dir>/_meta.json` (written by `agentcap
run`); pass `--agent <name>` if you're exporting traces captured
outside the orchestrator. Consumers read the union via
`load_dataset("hf://buckets/.../my-prefix/")`.

Dataset repos aren't a `--push` target on purpose: their semantics
are *atomic replace*, which doesn't fit a corpus that grows over
time. To publish a curated cut to a Dataset repo, render to
`--output` first and `hf upload` it yourself.

## What lands on disk

Per chat-completion request, two files in `<trace-dir>/`:
`<request_id>.request.json` (raw OpenAI request body) and
`<request_id>.response.json` (response body, or assembled stream
bytes for streaming). No tokenisation, no rendering, no derived
metadata — just the bytes.

## What the export layer adds

Each manifest row carries the raw `request`/`response` plus
chat-template-rendered token boundaries and per-message structural
metadata. The exact shape is whatever `agentcap.manifest.build_manifest`
returns — read that function rather than relying on a doc copy.

## Server backends

The proxy is backend-agnostic. Document the menu so users pick the
right one per use case:

| backend | when to use |
|---|---|
| **Inference Providers** (`router.huggingface.co/v1`) | demos, casual capture; zero infra; curated model catalogue; pay per token |
| **Inference Endpoints** | dedicated GPU + specific model + scale-to-zero between corpus runs; OpenAI-compat by default |
| **Local `llama.cpp` server** (`./llama-server`) | full control over quant / chat template / sampler; required for research that depends on model-implementation detail (e.g. kv-cache-reuse splice work) |
| `transformers serve` | works for small models, awkward for big ones at long context |

For which (backend, model, agent) combinations have been validated
end-to-end, see [docs/tested-models-and-agents.md](docs/tested-models-and-agents.md).

## Roadmap

See [ROADMAP.md](ROADMAP.md) — sandbox, skill-injection as a
first-class concept, corpus authoring generators, inspector Space.

## Running tests

```bash
pip install -e '.[dev]'
pytest tests/
```

Default runs unit tests only. Driver↔agent integration tests in
[tests/test_drivers_live.py](tests/test_drivers_live.py) skip unless
their prerequisites are present.

### Live agent tests

Each test invokes a real agent CLI against a live model server and
asserts the side-effect (a docstring landing in `hello.py`). It
skips unless **both**:

1. **The agent binary is reachable** — on `$PATH`, or via
   `AGENTCAP_TEST_<AGENT>_BIN` (`AIDER`, `GOOSE`, `PI`, `OPENCODE`,
   `HERMES`). Hermes also requires a populated `~/.hermes/config.yaml`.
2. **A model server is reachable** — either
   `AGENTCAP_TEST_LLM_URL=http://host:port/v1` (existing server) or
   `AGENTCAP_TEST_GGUF=/path/to/model.gguf` with `llama-server`
   resolvable (set `AGENTCAP_TEST_LLAMA_BIN` since `llama-server` is
   typically not on `$PATH`). The latter auto-spawns a
   session-scoped `llama-server` and tears it down on exit.

| env | default | purpose |
|---|---|---|
| `AGENTCAP_TEST_MODEL`    | `qwen3.6-35b-a3b` | model alias agents send |
| `AGENTCAP_TEST_NGL`      | `999`             | `--n-gpu-layers`; set `0` for CPU |
| `AGENTCAP_TEST_CTX_SIZE` | `8192`            | llama-server `--ctx-size` |

Each live test retries `drv.start()` up to 3 times to absorb
small-model sampling variance. For hard-green CI without live,
use `pytest -m "not live"`.

## License

Apache 2.0 — see [LICENSE](LICENSE).
