# agentcap

An end-to-end harness for running real coding agents at scale across
`(agent × model × corpus)` and publishing every interaction as a
[Hugging Face dataset](https://huggingface.co/docs/datasets). Drives
the agent through a corpus of prompts, captures every chat-completion
request/response from the wire (request bodies as parsed JSON;
streamed responses as raw SSE bytes), and pushes the result to the
Hub — so consumers can replay, render, or analyse what the agent
actually sent and got back, without reconstructing it from a log.

The loop:

```
  corpus  ──►  sandboxed agent run  ──►  capture  ──►  export  ──►  publish  ──►  inspect / replay
   ▲                                                                                         │
   └──────────────────── reuse for the next (agent, model) ──────────────────────────────────┘
```

## What this repo provides

- **Run agents through a corpus** — `agentcap run` drives one of the
  registered coding-agent CLIs (`hermes`, `opencode`, `goose`, `pi`)
  through a `tasks.txt`, inside a per-agent sandbox (bwrap on Linux,
  lima on macOS). Multi-turn follow-ups, optional skill injection.
- **Capture every wire interaction** — an in-process OpenAI-compat
  proxy sits between the agent and any backend that speaks
  `/v1/chat/completions`(llama.app, Inference Providers, vLLM).
  Request bodies are persisted as parsed JSON (the
  object, not the original byte sequence); streamed responses keep
  the raw SSE bytes. No tokenisation, no rendering — just persist
  what crossed the wire.
- **Keep the agent's own session log** — alongside captures, agentcap
  collects each agent's native trace (opencode's SQLite store, pi's
  JSONL stream, …) so consumers see both the agent's view and the
  wire view of the same run.
- **Publish to the Hub** — `agentcap export` bundles captures into
  parquet, ships the native traces alongside, and groups both as a
  Collection. Secret-scanned before push.
- **Inspect and replay** — `agentcap inspect` is an fzf-driven picker
  over runs and captures with a body preview; `agentcap replay <rid>`
  re-issues any captured request against any OpenAI-compatible target.

## Quick start

Install the sandbox prereqs (one-time) and agentcap itself.

```bash
# macOS
brew install lima

# Linux
sudo apt install -y bubblewrap buildah
# Ubuntu 24.04+ only: unprivileged user namespaces for bwrap.
sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0
echo 'kernel.apparmor_restrict_unprivileged_userns=0' \
    | sudo tee /etc/sysctl.d/60-agentcap-bwrap.conf

# Both
pip install -e .
```

Pick a server. Two flavours, same proxy front-end.

(a) Inference Providers `--upstream https://router.huggingface.co`
(b) Local inference server (like llama.app) `--upstream http://127.0.0.1:8000`

Example with a local `llama.app` server

```bash
./scripts/start_llama_cpp_server.sh ggml-org/gemma-4-E4B-it-GGUF &
```

Drive an agent through a corpus. Each run mints a fresh subdir under .agentcap/ in the workspace ($AGENTCAP_WORKSPACE or cwd).

```bash
agentcap run \
    --agent hermes \
    --model google/gemma-4-E4B-it \
    --upstream http://127.0.0.1:8000 \
    --tasks examples/transformers-coding-session/tasks.txt \
    --turns 4 --followup synthesized
```

Note: For `agentcap run`, `--model` is required for all agents.

Browse what's captured. ``--long`` adds upstream + per-run counts.

```bash
agentcap ls
```

Push everything. ``--push <owner>/<base>`` produces paired captures + traces
datasets and groups them in a Collection.

```bash
agentcap export --all --push my-org/my-captures
```

Or push selected runs only.

```bash
agentcap export hermes-local-20260512-162345 \
    --push my-org/my-captures
```

Browse captured requests

```bash
agentcap inspect                       # everything in the workspace
agentcap inspect <run-id>              # one run only
agentcap inspect <request-id>          # dump a specific body
```

Re-issue a single captured request to an OpenAI-compatible target.

```bash
agentcap replay <request-id> --target http://127.0.0.1:8000
```

See [docs/tested-models-and-agents.md](docs/tested-models-and-agents.md)
for which model + agent combinations have been validated end-to-end.

## Vocabulary

These terms appear throughout the CLI, on-disk layout, and docs:

| term | meaning |
|---|---|
| **task** | One corpus entry — the initial user prompt fed to the agent. |
| **turn** | One user-prompt cycle: either the initial prompt or one follow-up. Set by `agentcap run --turns N`. |
| **call** | One `/v1/chat/completions` request the agent makes to the model = one captured `<rid>`. A single turn contains many calls (the agent's tool-use loop). |
| **session** | An agent's stateful conversation. One task = one session, kept across all its turns via the agent's own `session_id`. |
| **capture** | One persisted `<rid>.request.json` + `<rid>.response.json` pair — one call's wire bytes on disk. |
| **trace** | The agent's own native session log for one task (per-agent format: opencode's SQLite, pi's JSONL, hermes' SQLite, …). |
| **run** | One `agentcap run` invocation: many tasks × many turns producing many calls, all under `.agentcap/<agent>-<provider>-<utc>/`. |

So a `run` contains N `task`s; each `task` is one `session` and runs over T `turn`s; each `turn` produces C `call`s, each `call` is one `capture`, and each `task` produces one `trace`.

## Architecture

```
  ┌─────────────────────────── runner ───────────────────────────────────┐
  │                                                                      │
  │  corpus ──► [agent CLI inside sandbox] ──HTTP──► [capture proxy] ──┐ │
  │       ▲             │              │                    │           ▼│
  │       │             ▼              ▼                    ▼      [model│
  │       │   final response text   <run>/traces/   <run>/captures/  server]
  │       │             │           (native log)    *.{req,resp}.json ▲  │
  │       │             ▼                                              │  │
  │       └── [follow-up synthesizer] ─────── HTTP (bypasses proxy)────┘  │
  │             (multi-turn, optional)                                    │
  └───────────────────────────────────────────────────────────────────────┘
                                       ▼
                               agentcap export
                                       │
                          ┌────────────┴────────────┐
                          ▼                         ▼
                <base>-captures dataset    <base>-<agent>-traces dataset
                          └────────────┬────────────┘
                                       ▼
                              Hub Collection
```

The synthesizer talks to the model server **directly**, around the
capture proxy, so the capture stays a clean record of agent ↔ model
interaction; the synthesizer's own LLM calls are an orchestration
detail and never land in the dataset.

Capture and export are deliberately dumb: persist the wire content,
ship it as parquet, stamp a few constant provider columns. Anything
token-level (chat-template rendering, per-message ranges) is
consumer-side — the `request` body is preserved as parsed JSON with
no agentcap-side normalisation of keys or values, so re-rendering
through the model's chat template is a few lines via
`transformers.AutoTokenizer.apply_chat_template`.

## Pushing to a Dataset repo

`agentcap export --push <owner>/<base>` walks one or more runs and
uploads, in a single commit each:

- `<owner>/<base>-captures` — one parquet per run, one row per
  chat-completion request. Filenames embed `(agent, model, provider)`
  so a single repo holds many tuples without aliasing. Consumers
  read the union via `load_dataset("<owner>/<base>-captures")`.
- `<owner>/<base>-<agent>-traces` — one repo **per agent**, holding
  that agent's native session-log files (opencode SQLite, pi JSONL,
  hermes SQLite, …). Only created for agents that produced any
  traces in the exported runs.

Both repos are added to a Collection titled `<base>` under `<owner>`
so they surface together on the Hub. On the first push to an empty
captures repo, agentcap also seeds a dataset card; subsequent pushes
leave any existing card untouched.

```bash
# Push every run in the workspace.
agentcap export --all --push my-org/my-captures

# Or push selected runs (run-ids from `agentcap ls`).
agentcap export hermes-local-20260512-162345 goose-local-20260512-170000 \
    --push my-org/my-captures

# Or point at an arbitrary workdir / capture dir directly.
agentcap export ./some/workdir --push my-org/my-captures
```

Before pushing, `agentcap export` runs `trufflehog` against each run
directory and aborts on any **verified** secret hit (pattern-only hits
are surfaced but don't block). Pass `--no-scan` to skip this gate.

## What lands on disk

Each `agentcap run` invocation creates one directory at
`$AGENTCAP_WORKSPACE/.agentcap/<agent>-<provider>-<utc>/` (or
`./.agentcap/...` when `AGENTCAP_WORKSPACE` is unset). Inside it:

- `run.json` — run-level metadata (agent, model, provider, upstream,
  task list, per-task durations). Updated at end-of-run; a stub
  version is written at start so `agentcap ls / inspect` can see
  in-flight runs.
- `captures/` — one `<rid>.request.json` + `<rid>.response.json`
  pair per chat-completion call. Request bodies are stored as parsed
  JSON (the object, not the original byte sequence); streamed
  responses keep the raw SSE bytes verbatim.
- `traces/` — the agent's own native session log (opencode and hermes
  drop a SQLite dump per session; pi streams JSONL; goose writes its
  built-in log file). Format is per-agent; agentcap is just a courier.
- `sessions/` — the orchestrator's per-turn `stdout` / `stderr` from
  the agent CLI, one file pair per `<task_id>_turn_<NN>`.
- `sandbox/` — the agent's cwd inside the sandbox (bind-mounted from
  here so anything the agent writes survives the run).

Captures are dumb: no tokenisation, no rendering, no derived metadata
— just the bytes that crossed the proxy.

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
fields. Consumers `json.loads` them.

## Server backends

The proxy is backend-agnostic. Pick whichever fits the use case:

| backend | when to use |
|---|---|
| **Inference Providers** (`router.huggingface.co`) | demos, casual capture; zero infra; curated model catalogue; pay per token |
| **Local `llama.app` server** (`llama serve`) | full control over quant / chat template / sampler; required for research that depends on model-implementation detail (e.g. kv-cache-reuse splice work) |
| `transformers serve` | works for small models, awkward for big ones at long context |

For which (backend, model, agent) combinations have been validated
end-to-end, see [docs/tested-models-and-agents.md](docs/tested-models-and-agents.md).

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
Override the agent's model id with `AGENTCAP_TEST_MODEL` and the
GGUF with `AGENTCAP_TEST_GGUF` (defaults to Qwen3-1.7B-Q8 fetched
from the Hub — small + fast enough to chain a tool call on CPU,
which is what the live tests assert).

The per-agent sandbox is built / booted lazily on first use (same
lifecycle as `agentcap run`), so the first session pays a multi-
minute cold-build per agent.

## License

Apache 2.0 — see [LICENSE](LICENSE).
