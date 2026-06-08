# Capturing an agent run

`agentcap run` drives a registered coding-agent CLI (`hermes`,
`opencode`, `goose`, `pi`) through a list of tasks, inside a per-agent
podman sandbox. Every `/v1/chat/completions` request the agent makes
is recorded by the in-process capture proxy.

![capture demo](img/capture.gif)

_Regenerate from the repo root with `vhs docs/demo/capture.tape` —
the tape's ``Output`` directive writes ``docs/img/capture.gif``._

## Minimum command

```bash
agentcap run \
    --agent hermes \
    --model google/gemma-4-E4B-it \
    --upstream http://127.0.0.1:8000 \
    --tasks examples/transformers-coding-session/tasks.txt
```

Four required pieces:

| flag        | what it picks                                                |
|-------------|--------------------------------------------------------------|
| `--agent`   | which CLI to drive (`hermes` \| `opencode` \| `goose` \| `pi`) |
| `--model`   | the model id the agent will ask for (required for all agents)  |
| `--upstream`| where the proxy forwards calls — any OpenAI-compat endpoint  |
| `--tasks`   | path to `tasks.txt`, one initial user prompt per line        |

The proxy listens on a free local port and rewrites the agent's
endpoint to point at itself, so the agent talks to the proxy and the
proxy talks to `--upstream`. The agent never sees the real upstream.

## Multi-turn + follow-ups

`--turns N` runs the agent on the same task `N` times. Turn 1 is the
prompt from `tasks.txt`; turns 2..N use a follow-up strategy:

```bash
agentcap run --agent goose --model zai-org/GLM-4.6 \
    --upstream https://router.huggingface.co \
    --tasks examples/hf-hub-session/tasks.txt \
    --turns 4 --followup synthesized
```

| `--followup` | next-turn prompt                                          |
|--------------|-----------------------------------------------------------|
| `continue`   | literal string `"continue"`                                |
| `templates`  | one of a fixed pool (varies per turn)                      |
| `synthesized`| a small LLM is asked to produce a natural next user message |

`synthesized` calls a separate model — by default the same upstream,
override with `--synth-upstream` / `--synth-model`. The follow-up call
**bypasses the capture proxy** so the capture stays a clean record of
agent ↔ model interaction.

## Where captures land

Each invocation creates one directory under
`$AGENTCAP_WORKSPACE/.agentcap/` (or `./.agentcap/` if the env var is
unset):

```
.agentcap/hermes-local-20260605-122521/
├── run.json                              # run-level metadata
├── captures/                             # one rid per chat-completion call
│   ├── 6437573b....request.json
│   ├── 6437573b....response.json
│   └── …
├── traces/                               # agent's own native session log
│   ├── task_01-session.sqlite
│   └── …
├── sessions/                             # per-turn stdout/stderr
│   ├── task_01_turn_01.stdout.log
│   └── …
└── sandbox/                              # agent's cwd inside the container
```

## Sandbox prereqs

`agentcap run` always runs the agent inside podman. Once-per-machine
setup:

```bash
# macOS
brew install podman
podman machine init --memory 8192   # 4 GB is the minimum for the test GGUF
podman machine start

# Linux
sudo apt install -y podman
```

The per-agent container image is built lazily on first use — expect a
multi-minute cold build the first time each agent is invoked.

## Server backends

The proxy is backend-agnostic; `--upstream` is the only switch.

| backend                       | when to use                                          |
|-------------------------------|------------------------------------------------------|
| Inference Providers (`router.huggingface.co`) | demos, casual capture; pay-per-token         |
| Local `llama.app` server      | full control over quant / chat template / sampler    |
| `transformers serve`          | small models; awkward for big ones at long context   |

For known-good `(backend, model, agent)` tuples see
[docs/tested-models-and-agents.md](tested-models-and-agents.md).

## What to do next

- `agentcap ls` — list runs in the workspace.
- [docs/inspect.md](inspect.md) — browse captured requests.
- [docs/replay.md](replay.md) — re-issue a captured request elsewhere.
- [docs/export.md](export.md) — publish to a HF dataset.
