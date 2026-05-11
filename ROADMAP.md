# Roadmap

The repo started as a capture proxy. It is now an end-to-end harness
for running coding agents at scale across `(agent × model × corpus)`
and publishing every interaction as a reproducible dataset. Capture
is one stage; the loop is **corpus → sandboxed run → capture →
export → publish → inspect**.

The items below are ordered roughly by load-bearing-ness.

## Framing

- **Rename open.** `agentcap` no longer covers what's here. Candidate:
  `agentbed` (a testbed for running coding agents at scale). The cost
  of a rename is real — links, the bucket prefix
  `dacorvo/agentcap-traces`, the `import agentcap` consumers,
  references from kv-reuse-lab. Decide before any further
  README rewrite.
- **README rewrite** to lead with the workflow, not capture. Proxy
  becomes a "how it works" subsection, not the headline.

## Producer side

### Sandbox

Every agent run executes in a controlled environment so pi-style
absolute-path file writes can't leak into the host repo.

- `bwrap` (bubblewrap) on Linux, `lima` on macOS.
- Mount: the per-run sandbox dir read-write, everything else
  read-only or hidden.
- Wraps the existing per-agent subprocess invocation — driver code
  stays the same except for the launch.
- In flight.

### Skill injection as a first-class concept

Today `examples/hf-hub-session/run.sh` does the wiring inline (install
HF skills, seed `AGENTS.md`, symlink `~/.agents/skills/`). Lift it out
into `agentcap run --skills <pkg-or-path>`:

- For `hermes`: install into the `~/.hermes/skills/` overlay.
- For `opencode` / `goose` / `pi`: seed `AGENTS.md` + a `skills/`
  symlink into the sandbox.
- `<pkg>` resolves to entries in the
  [`huggingface/skills`](https://github.com/huggingface/skills)
  marketplace via `hf skills`; `<path>` to a local SKILL.md
  directory.

### Server backends — pick the right one per use case

The proxy is backend-agnostic. Document the menu:

| backend | when to use |
|---|---|
| **Inference Providers** (`router.huggingface.co/v1`) | demos, casual capture; zero infra; curated model catalogue; pay per token |
| **Inference Endpoints** | dedicated GPU + specific model + scale-to-zero between corpus runs; OpenAI-compat by default |
| **Local llama-server** | kv-reuse-lab research and anything that needs control over quant / chat template / sampler |

Optional ergonomic wrapper: `agentcap endpoints {create,delete}` that
spins an Inference Endpoint for a corpus run and tears it down on
exit, so a user without local GPUs can do `agentcap run --upstream
$(agentcap endpoints create --model X --hardware A10G)`.

HF Jobs is **not** a fit for the server role: it's batch-script
shaped, mismatches a long-running serve loop. Could still be used to
wrap the whole `run + export` as a parameterised cloud job, but
that's secondary.

### Corpus authoring

Today's corpora (`examples/transformers-coding-session/tasks.txt`,
`examples/hf-hub-session/tasks.txt`) are hand-written. Real workloads
expose patterns hand-written prompts miss — and for kv-reuse-lab in
particular, the recurrence structure that the cache exploits only
shows up in real prompts.

A small `agentcap corpus <source>` family emits `tasks.txt`:

- `agentcap corpus from-issues <gh-repo> [--label X] [--state open|closed] [--limit N]`
  — MVP. Pulls real GitHub issues, formats title + body into agent
  prompts. Built on `gh api`.
- `agentcap corpus from-prs <gh-repo>` — same but for PR descriptions.
- `agentcap corpus from-forum <discourse-url> [--category X]` —
  Discourse forum threads (`discuss.huggingface.co` is the obvious
  source for hf-hub corpora).
- `agentcap corpus from-prompt "<meta-prompt>" --model <id> --n 30`
  — fallback. Uses a small LLM to draft prompts in the corpus
  style; bypasses the capture proxy.

## Consumer side

### Inspector Space

Public Gradio CPU Space, pure parquet reader. Loads via
`load_dataset("hf://buckets/.../<prefix>/")`. Three views:

1. **Session timeline** — chat-style thread, system prompt collapsed,
   tool calls as expandable cards showing `(tool_name, arguments,
   result)`. The front door.
2. **Rendered-bytes pane** — the captured prompt as the model actually
   received it, colour-coded per role using `sections` + `token_role`
   from the manifest. The differentiator: no other agent dashboard
   shows the byte stream; agentcap can.
3. **Cross-cell comparison** — pick the same task across two
   `(agent, model)` parquets, see how the tool-call pattern, turn
   count, and token spend differ.

Stretch:
- Stable-prefix highlight on the rendered-bytes pane — ties directly
  to kv-reuse-lab's recurrence story.
- Filter: "all sessions where tool X was called with args matching Y"
  — exposes the recurrence structure that drives the kv-reuse use
  case.

## Out of scope

- **Replay.** Re-issuing captured requests against a different model
  is per-request only — the agent's next prompt diverges the moment
  the model responds differently. The only useful per-request replay
  is kv-reuse-lab's existing splice-correctness harness, which
  doesn't need agentcap support.
