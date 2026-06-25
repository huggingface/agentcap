# agentcap

An end-to-end harness for running real coding agents at scale across
`(agent × model × corpus)` and publishing every interaction as a
[Hugging Face dataset](https://huggingface.co/docs/datasets). Drives
the agent through a corpus of prompts, captures every chat-completion
request/response from the wire (request bodies as parsed JSON;
streamed responses as raw SSE bytes), and pushes the result to the
Hub — so consumers can render, analyse, or re-issue what the agent
actually sent and got back, without reconstructing it from a log.

The pipeline:

```
  corpus  ──►  sandboxed agent run  ──►  capture  ──►  export  ──►  publish  ──►  inspect
```

Repeat for each `(agent, model)` you want compared — the corpus
stays the same.

![inspect demo](docs/img/inspect.gif)

_Three-level picker chain over an HF dataset of captures
(`hf://datasets/<owner>/<name>`): parquet → request → message,
with live preview and Esc walk-back. See
[docs/inspect.md](docs/inspect.md) for the rest._

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/huggingface/agentcap/main/scripts/install.sh | sh
```

[`scripts/install.sh`](scripts/install.sh) detects your platform, downloads the
matching binary, and installs it to `~/.local/bin`. Pass flags after `sh -s --` —
`-b <dir>` to change the install dir, `-v <tag>` to pin a version:

```bash
curl -fsSL https://raw.githubusercontent.com/huggingface/agentcap/main/scripts/install.sh | sh -s -- -b /usr/local/bin
```

Or grab a binary straight from [GitHub Releases](https://github.com/huggingface/agentcap/releases):

| Platform            | Binary                        |
|---------------------|-------------------------------|
| Linux x86_64        | `agentcap-x86_64-linux`       |
| macOS Apple Silicon | `agentcap-arm64-apple-darwin` |

To build it yourself instead, see [Building from source](#building-from-source).

## Quick start

agentcap shells out to two external tools at runtime: `podman` runs the
per-agent sandbox (`agentcap run`), and `trufflehog` runs the pre-push
secret scan (`agentcap export` aborts on any verified hit; pass
`--no-scan` to skip). The inspect pickers are built in — no fzf.

```bash
# macOS
brew install podman trufflehog
podman machine init --memory 8192    # one-time; needs ≥4 GB for the test GGUF
podman machine start

# Linux
sudo apt install -y podman
curl -sSfL https://raw.githubusercontent.com/trufflesecurity/trufflehog/main/scripts/install.sh \
    | sh -s -- -b ~/.local/bin
```

Pick a server — typically (a) Inference Providers
`--upstream https://router.huggingface.co` or (b) a local OpenAI-compat
server like `llama.app` on `http://127.0.0.1:8000`. See
[docs/capture.md](docs/capture.md) for the trade-offs.

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

## Usage

The three sub-commands have a dedicated walkthrough each — flags,
flows, and a recorded demo:

| command           | docs page                                  |
|-------------------|--------------------------------------------|
| `agentcap run`    | [docs/capture.md](docs/capture.md) — sandboxes, multi-turn, follow-ups, backends |
| `agentcap inspect`| [docs/inspect.md](docs/inspect.md) — workspace / parquet / HF dataset pickers   |
| `agentcap export` | [docs/export.md](docs/export.md) — push captures + traces as a HF Collection  |

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

## Building from source

A Rust toolchain is all you need — no system libraries. The version is
pinned in [rust-toolchain.toml](rust-toolchain.toml) (rustup fetches it
automatically).

```bash
cargo build --release      # binary at target/release/agentcap
cargo install --path .     # …or install it onto your PATH (~/.cargo/bin)
```

## Running tests

```bash
cargo test                 # unit + loopback proxy integration (hermetic; no podman)
```

The **live** tier drives the real `agentcap run` binary through a model
server for each agent inside podman, so it's `#[ignore]`d by default. With
a server reachable — `AGENTCAP_TEST_LLM_URL=http://host:port`, or one
already on `:8000` / `:8080`:

```bash
cargo test --test live -- --ignored
```

Each live test skips (passes) if no server is reachable; override the model
id with `AGENTCAP_TEST_MODEL`. The per-agent sandbox image is built lazily
on first use (same lifecycle as `agentcap run`), so the first run pays a
multi-minute cold build per agent. CI runs this as the **Test - Live**
workflow against a pinned llama.cpp + Qwen3-1.7B GGUF.

## License

Apache 2.0 — see [LICENSE](LICENSE).
