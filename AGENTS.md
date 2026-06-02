# agentcap — agent handoff notes

Read this before changing the code. The README explains what the
project is and how to use it; this file holds **decisions that already
hardened** and **what's not yet built**.

## Architecture decisions — DO NOT relitigate

These are settled. If you want to revisit any, raise it with the user
explicitly first.

1. **Capture is dumb.** The proxy persists raw OpenAI-compat bytes.
   No tokenisation, no chat-template render at capture time. The
   capture path must work without the model's tokenizer being loaded.

2. **Export is a dumb data shuffle.** No tokenizer, no chat-template
   render, no per-token labels. Export pairs `<rid>.request.json`
   with `<rid>.response.json`, serialises each as a JSON string into
   a parquet row, and stamps a couple of constant provider columns
   plus per-row `served_by` / `served_build_info` / `served_model`
   from the response fingerprint. The capture path doesn't need the
   model's tokenizer loaded; neither does the export path.

3. **No hashable identifiers, no rendered tokens in rows.** The
   parquet exposes the raw `request` and `response` bodies as JSON
   strings. Token-level analysis is consumer-side: render
   `request.messages` through the model's chat template via
   `transformers.AutoTokenizer.apply_chat_template` and compute
   whatever ids you want. Reasons:
   - Different consumers want different definitions of "prefix id",
     "args hash", "agent build id". Shipping one in rows means
     everyone computes both ours and theirs.
   - Some templates (Qwen3-Coder, Gemma-4) crash on list-typed
     `content` or string-typed `tool_call.arguments` from real
     captures and need normalisation before render. Owning that
     normalisation in the producer locks consumers into our exact
     normaliser; leaving render consumer-side lets each consumer
     handle template quirks on their own terms.

4. **Capture is via a transparent HTTP proxy, not via patches to the
   serving stack.** Compatibility with any OpenAI-compat backend is
   load-bearing. No fork patches, no in-server hooks.

5. **Upstream is a static base URL.** Configured at proxy startup;
   path is mirrored verbatim onto upstream. No upstream pool, no
   per-request routing.

6. **Only `/v1/chat/completions` POST is captured.** All other paths
   pass through transparently with no capture files.

7. **Streaming responses: forward chunk-by-chunk, persist the
   assembled raw bytes at end-of-stream.** SSE parsing into discrete
   events is the export layer's job, not the proxy's.

8. **Synthesised follow-ups bypass the capture proxy** — they go
   straight to the model server so the captured corpus stays a clean
   record of agent↔model interaction.

9. **`HermesDriver` builds a sandboxed `HERMES_HOME` overlay inside
   the sandbox** so a capture run never reads or mutates the
   sandbox-side `~/.hermes/` state. The overlay is materialised
   entirely via the Sandbox protocol (`sandbox.mkdtemp` then `sh -c
   "cp -aL ~/.hermes/. <overlay>"` then `rm -rf` of
   `_HERMES_FRESH_PER_RUN` entries then `sandbox.write_text` of the
   rewritten `config.yaml`). On Linux/bwrap the sandbox-side
   `~/.hermes/` is the host's; on Lima it's the VM-provisioned
   home inside the `agentcap-hermes` VM. Either way the user's
   actual home is never touched.

   - **Snapshot**: identity content (`skills/`, `SOUL.md`,
     `hermes-agent/`, `hooks/`, `pairing/`, `models_dev_cache.json`)
     is brought across by the `cp -aL`. Agent writes inside the
     overlay diverge from the source.
   - **Fresh per-run** (`_HERMES_FRESH_PER_RUN`): `memories/`,
     `sessions/`, `sandboxes/`, `state.db`, `logs/`, `cron/`,
     `image_cache/`, `audio_cache/`, `auth.lock`. Wiped from the
     snapshot after the copy and recreated empty (files are left
     absent so hermes recreates them on demand). Discarded on
     driver close.

   `config.yaml` is regenerated with the proxy `base_url` swapped
   in and (optionally) the two `context_length` guards lowered for
   CPU/small-model runs.

   Skills used by the corpus (e.g. `huggingface/skills` for the
   `hf-hub-session` runs) are injected per-run via
   `agentcap run --skills <dir>`. The runner bind-mounts the dir
   read-only into the sandbox and exposes it as
   `AGENTCAP_SKILLS_DIR`; the per-agent image entrypoint symlinks
   it into the agent's discovery path (`~/.hermes/skills/` for
   hermes; `AGENTS.md` + `skills/` in cwd for opencode/goose/pi).

   **Lifecycle is per-`agentcap run` invocation, not per-task.** The
   overlay is built once when the driver is constructed and reused
   for every prompt in the corpus. Memory written in task 1 is
   visible in the system prompt of task 2; the agent's `state.db`
   accumulates across all tasks in the run. This is intentional: it
   matches what a real user experiences when they invoke
   `hermes chat` repeatedly — each invocation reads and writes the
   same persistent home — so the captured corpus reflects realistic
   cross-invocation memory evolution. The trade-off, called out in
   the original bug report, is that the `MEMORY` section of the
   system prompt grows turn-over-turn and task-over-task within a
   run, shifting the prefix in ways consumers must handle. Two
   separate `agentcap run` invocations both start from empty memory,
   so per-invocation reproducibility is preserved.

10. **Hermes runs from a clean per-run sandbox cwd**
    (`<workdir>/sandbox/`), not from agentcap's invocation cwd.
    Hermes auto-injects `AGENTS.md` / `CLAUDE.md` / `.cursorrules`
    from the cwd into every system prompt; running from a project
    dir leaks those files into every capture, contaminating
    the dataset's "stable" prefix.

11. **`--push` writes to Hugging Face Dataset repos.** One
    `agentcap export` invocation produces one git commit, regardless
    of how many runs it bundles; files land under
    `data/[<subdir>/]<file>.parquet`. Repos are auto-created on
    first push, and a starter dataset card is seeded then (left
    alone on subsequent pushes). Buckets were the original target;
    the move to Datasets was driven by the free Hub Dataset Viewer
    that lights up automatically on `data/*.parquet`. The
    "atomic-replace" semantics of `Dataset.push_to_hub` would
    overwrite a growing corpus, so agentcap doesn't use it.

12. **Each `push_dataset` call writes a unique parquet filename by
    default** that embeds `(agent, model, provider)` so the filename
    alone tells you what's inside —
    `train-<agent>-<model>-<provider>-YYYYMMDDTHHMMSS-HEX6.parquet`.
    Each part is optional and is omitted when unknown; `agent` is
    read from the run's `run.json`, `model` and `provider` are
    derived from the captured requests. An explicit `filename=` opts
    back into overwrite-in-place — used only for "latest" pointer
    files.

13. **One output format only: parquet.** Single file per run, pushed
    via `--push`. JSONL
    was dropped — it's a one-liner away from a parquet via
    `Dataset.from_parquet(...).to_json(...)`. Rendered token ids and
    per-message structural metadata are likewise consumer-side: a
    5-line recompute via `apply_chat_template` keeps rows small and
    avoids pinning consumers to our exact normalisation of the
    template-input shape (see decision 3).

14. **Replay applies no agentcap-side normalisation.** `agentcap
    replay` re-issues a captured request with no flags that mutate
    the body. The request is persisted as parsed JSON (so the
    original byte sequence — whitespace, key ordering — isn't
    recoverable, only the JSON object); streamed SSE response bytes
    are kept verbatim. Cross-server strictness asymmetries (e.g.
    captures from a lenient upstream sent at a strict upstream
    that rejects explicit `null`s) are the consumer's normalisation
    problem, not agentcap's. Multi-turn replay stays out of scope
    because conversation state diverges as soon as the new model
    responds differently.

15. **Inference backend must deliver tool calls in `message.content`,
    not `message.reasoning_content`.** Hermes (and presumably other
    agents) parses tool calls from the OpenAI-spec `content` field.
    Reasoning-by-default models (Qwen 3.5+, etc.) on llama.cpp put
    their actual answer in `reasoning_content` and leave `content`
    empty — the agent loop sees no tool calls and stalls. Run
    `llama serve` with `--reasoning off` for these models;
    `scripts/start_llama_cpp_server.sh` exposes this via the
    `REASONING` env var (default `auto` follows the chat-template's
    own default, set to `off` for reasoning models). Also: agent
    capture requires a post-agentic-era model and a context window
    ≥64K tokens — Hermes refuses to launch otherwise.

## What's not yet built

1. **YAML tasks file format.** `read_tasks_txt` handles `.txt`. Soft-
   import PyYAML to keep it optional.

2. **`agentcap run` graceful interrupt.** Catch `KeyboardInterrupt`,
   finish the in-flight turn, write a partial `run.json`.

3. **vLLM backend smoke test.** llama.cpp is the validated default;
   verify the proxy stays transparent against vLLM too.

4. **Corpus-specific VM mounts.** With the default `mounts: []`,
   corpora that need specific host content inside the VM — e.g.
   `transformers-coding-session`'s transformers source tree — must
   either ship a corpus-specific Lima template variant (mounting
   that path read-only) or use `limactl edit <vm>` to amend the
   provisioned VM and restart it. Tokens / per-run secrets (e.g.
   `HF_TOKEN`) flow through `sandbox.run(env={…})` →
   `limactl shell -- env KEY=VAL …`, no mount required.
