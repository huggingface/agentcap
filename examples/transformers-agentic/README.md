# transformers-agentic

agentcap port of the [`is-it-agentic-enough`](https://github.com/huggingface/is-it-agentic-enough)
task suite (the [blog post](https://huggingface.co/blog/is-it-agentic-enough)):
16 prompts that each ask an agent to run a **named** Hugging Face model
(classify sentiment, transcribe audio, caption an image, …) and report the
result. Because each task pins a specific model, the agent has to actually
load and run it rather than answer from world knowledge.

Here it's used to **compare models/agents** through agentcap's capture path —
not to reproduce the article's scoring. agentcap records the agent ↔ model
wire traffic; match %, token, and CLI-vs-`pipeline()` marker analysis are the
upstream harness's job (the captures contain what's needed to compute them
later).

## How the agent actually runs transformers

The agent's task work executes **inside the podman sandbox**, which ships only
the agent CLI — no transformers. Rather than rebuild the images, a
self-contained, relocatable `transformers` bundle is mounted read-only via
`agentcap run --tool-dir` and put on the agent's PATH:

```bash
./build-toolenv.sh        # one-time: builds ./toolenv/ + prewarms the model cache
```

`build-toolenv.sh` builds the bundle **inside `ubuntu:24.04`** — the base of
every agentcap agent image — so the venv's interpreter and torch `.so`s are
ABI-identical when mounted into any sandbox. It pins the exact transformers
commit that carries the (still unreleased) agentic CLI, installs CPU torch, and
prewarms every corpus model into `./toolenv/hf-cache/`. The venv configures
itself to use that cache — a `.pth` points `HF_HOME` at it (resolved from the
venv root, so it holds wherever the bundle is mounted) and defaults to offline —
so runs read models from the read-only mount with no network or re-downloads.

## Tiers (the article's discovery conditions)

| `--tier` | what the agent gets |
|---|---|
| `bare`  | empty cwd; only the mounted `transformers` bundle |
| `clone` | cwd is a git worktree of `./transformers` @ the bundle's commit (AGENTS.md / `cli/agentic/*.py` auto-discover) |
| `skill` | empty cwd + the packaged transformers Skill (`./skill`) in context |

## Run

```bash
# server: any OpenAI-compat /v1 on $UPSTREAM (default http://127.0.0.1:8001)
./run.sh --agent pi     --model unsloth/GLM-4.5-Air-GGUF --tier skill
./run.sh --agent hermes --model unsloth/GLM-4.5-Air-GGUF --tier bare
```

`./run.sh --help` for the env knobs. It pins `AGENTCAP_WORKSPACE` here, so runs
live under `./.agentcap/` — list them with `agentcap ls` from this directory, and
publish with `agentcap export <run-id|--all> --push <owner>/<dataset>`.

`tasks.txt` is the full 16-task corpus; pass `--tasks <file>` to run a subset.

## Caveats vs. the article

- **One cwd per (agent, model, tier) run**, reused across the corpus's tasks
  (agentcap runs a corpus in a single sandbox), where the article isolates each
  task in its own worktree. File writes from one task can persist into the next.
- The agentic CLI is unreleased; the bundle pins commit
  `4d15b215f3` (`is-it-agentic-enough`'s "w/ CLI + Skill" ref).
- Prewarm uses the classic HTTPS backend (`HF_HUB_DISABLE_XET=1`): xet stalled
  once on a transient CAS hiccup during a long bulk download, and HTTPS is
  steadier for a one-shot prewarm. (xet into the bind-mounted cache itself works
  fine — verified; it's not a mount problem.) Runs read the cache offline, so
  xet is never invoked at run time regardless.
