# agentcap

A framework for capturing real LLM-agent chat-completion traffic and
exporting it as a [Hugging Face dataset](https://huggingface.co/docs/datasets).
What an agent actually sends to a model — its full system prompt,
tool schemas, message history, tool calls, and tool results, every
turn — preserved verbatim and indexed by `(model, agent build)`.

The output is a clean dataset other people can load and study without
re-running the agent. Useful for evaluation, fine-tuning corpora,
behaviour analysis, debugging an agent's prompt assembly, and
prefix-cache strategy design — anything that needs the actual byte
stream the agent sent, not a reconstructed approximation.

## What this repo provides

Three components, each independently useful:

1. **Capture proxy** (`agentcap proxy`) — a transparent OpenAI-compat
   HTTP proxy between an agent CLI and a model server, dumping every
   request/response pair to disk. Backend-agnostic. Usable
   independently if you want to drive the agent yourself.
2. **Orchestrator** (`agentcap run`) — drives a list of prompts
   through a real agent CLI, starts the proxy in-process, and
   optionally extends each session with multi-turn follow-ups.
   `agentcap run --help` lists the supported agents (currently
   Hermes, OpenCode, Goose, pi-mono).
3. **Dataset export** (`agentcap export`) — rolls a captured trace
   dir into a Hugging Face dataset, with chat-template-rendered token
   boundaries and per-message structural metadata.

```
  ┌──────────────────────── orchestrator ──────────────────────┐
  │                                                            │
  │  task list ──► [agent CLI] ──HTTP──► [capture proxy] ──HTTP──► [model server]
  │       ▲             │                       │
  │       │             ▼                       ▼
  │       │     final response text     <trace-dir>/*.{request,response}.json
  │       │             │                       │
  │       │             ▼                       │
  │       └─── [follow-up synthesizer]          │
  │              (multi-turn)                   │
  └─────────────────────────────────────────────┘
                                                ▼
                                  agentcap export
                                                │
                                                ▼
                                       Hugging Face dataset
```

The split is intentional. **Capture is dumb** — no tokenizer, no
chat-template render, no per-token labels — just persist the bytes.
**Export is smart** — loads the model's tokenizer, renders, computes
per-message token ranges and role labels.

You can use any subset:
- All three: a one-command experiment runner.
- Proxy + export: drive the agent yourself, capture passively.
- Export alone: bring your own captured traces, get a clean dataset.

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

## Quick start

```bash
pip install -e .

# Your model server. llama.cpp is the recommended path — fast cold
# start, predictable memory, multi-GPU via tensor-split, GGUF quants
# fit large models on consumer hardware. transformers serve also
# works (and the proxy is backend-agnostic), but is harder to make
# fit at long context for big models.
GGUF_PATH=/path/to/model-Q4_K_M.gguf REASONING=off \
    ./scripts/start_llama_cpp_server.sh &

# Drive Hermes through 30 prompts × 4 turns, capture, then export
agentcap run \
    --agent hermes \
    --upstream http://127.0.0.1:8000 \
    --tasks examples/transformers-coding-session/tasks.txt \
    --turns 4 --followup continue \
    --workdir runs/run-001/

agentcap export runs/run-001/traces \
    --output runs/run-001.parquet
```

See [docs/tested-models-and-agents.md](docs/tested-models-and-agents.md)
for which model + agent combinations have been validated end-to-end.

`--model` is inferred from the captured request bodies; pass it
explicitly to override or when traces lack a model field. The trace
dir must be for a single model (the dataset format never mixes them).

`agentcap run` starts the capture proxy in-process and configures the
agent to talk through it. If you'd rather drive the agent yourself, run
`agentcap proxy` standalone and adjust your agent configuration to point
to the proxy.

### Pushing to a Storage Bucket

`--push` writes the parquet directly into a [Hugging Face Storage
Bucket](https://huggingface.co/docs/hub/storage-buckets) — mutable,
append-by-prefix, Xet-deduplicated:

```bash
agentcap export <trace-dir> --push hf://buckets/my-org/agentcap-traces/hermes-gemma-4-E4B-it/
```

Each run lands as a unique parquet file under the supplied prefix
(`train-YYYYMMDDTHHMMSS-HEX6.parquet` by default). Consumers read the
union via `load_dataset("hf://buckets/.../hermes-gemma-4-E4B-it/")`.

Dataset repos aren't a `--push` target on purpose: their semantics are
*atomic replace*, which doesn't fit a corpus that grows over time. To
publish a curated cut to a Dataset repo, render to `--output` first
and `hf upload` it yourself.

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

## Backend status

| backend | proxy compat | recommended |
|---|---|---|
| `llama.cpp` server (`./llama-server`) | ✓ | ✓ default — fast cold start, GGUF quants, predictable memory |
| `transformers serve` (Hugging Face) | ✓ | works for small models, awkward for big ones at long context |
| OpenAI / OpenRouter / hosted endpoints | ✓ | not the use case (this repo is for private-weights captures) |

For which (backend, model, agent) combinations have been validated
end-to-end, see [docs/tested-models-and-agents.md](docs/tested-models-and-agents.md).

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

#### CPU + small-model run

Validated against
[`unsloth/Qwen3-4B-Instruct-2507-GGUF`](https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF)
(Q4_K_M, ~2.5 GB) — completes in ~1 min on a 16-vCPU box:

```bash
AGENTCAP_TEST_GGUF=/path/to/Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
AGENTCAP_TEST_LLAMA_BIN=/path/to/llama-server \
AGENTCAP_TEST_NGL=0 \
AGENTCAP_TEST_CTX_SIZE=4096 \
AGENTCAP_TEST_MODEL=qwen3-4b \
pytest tests/test_drivers_live.py -v
```

Each live test retries `drv.start()` up to 3 times to absorb
small-model sampling variance. For hard-green CI without live,
use `pytest -m "not live"`.

## License

Apache 2.0 — see [LICENSE](LICENSE).
