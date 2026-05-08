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

2. **Manifest is offline and dumb.** All token-level metadata
   (`tok_range`, `tools_injection_tokens`, `token_role`) is computed
   at *export* time by re-rendering through the model's chat template.
   The runtime path produces no manifest sidecar.

3. **No derived metadata in manifest rows.** Rows expose the raw
   `request` body and structural `sections` only. Anything that
   reduces those to a single hashable identifier (prefix ids, args
   hashes, "agent build" ids) is a consumer-side choice — different
   consumers want different definitions, and shipping one in rows
   means everyone computes both ours and theirs.

4. **Capture is via a transparent HTTP proxy, not via patches to the
   serving stack.** Compatibility with any OpenAI-compat backend is
   load-bearing. No fork patches, no in-server hooks.

5. **Upstream is a static base URL.** Configured at proxy startup;
   path is mirrored verbatim onto upstream. No upstream pool, no
   per-request routing.

6. **Only `/v1/chat/completions` POST is captured.** All other paths
   pass through transparently with no trace files.

7. **Streaming responses: forward chunk-by-chunk, persist the
   assembled raw bytes at end-of-stream.** SSE parsing into discrete
   events is the export layer's job, not the proxy's.

8. **Synthesised follow-ups bypass the capture proxy** — they go
   straight to the model server so the captured corpus stays a clean
   record of agent↔model interaction.

9. **`HermesDriver` builds a sandboxed `HERMES_HOME` overlay** so a
   capture run never reads or mutates the user's persistent
   `~/.hermes/` state. Identity content (`config.yaml` rewritten to
   point at the proxy, `skills/`, `SOUL.md`, …) is symlinked through
   so the agent's behaviour is preserved. Writable per-run state
   (`memories/`, `sessions/`, `state.db`, `sandboxes/`, `logs/`,
   `cron/`) is fresh-empty in the overlay at start-of-run, and
   discarded on driver close. Without this, `memory(action=add)`
   calls during a run would write through into the user's real
   `~/.hermes/memories/MEMORY.md`.

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
    dir leaks those files into every captured trace, contaminating
    the dataset's "stable" prefix.

11. **`stable=True` only for leading system messages.** Once any
    non-system message appears, no later section is `stable`, even if
    a later message has `role=system`.

12. **`--push` only writes to Storage Buckets, never to Dataset
    repos.** Buckets are append-by-prefix — the natural shape for a
    corpus that grows. Dataset repos are atomic-replace via
    `push_to_hub`, which doesn't fit. To publish a curated cut to a
    Dataset repo, render to `--output` then `hf upload` it. The
    "load-concat-repush to fake append on a Dataset repo" pattern is
    explicitly rejected.

13. **Each `push_bucket` call writes a unique parquet filename by
    default** that embeds (agent, model) so the filename alone tells
    you what's inside —
    `train-<agent>-<model>-YYYYMMDDTHHMMSS-HEX6.parquet`. Agent is
    optional and falls through to `train-<model>-…` when absent
    (older trace dirs); the orchestrator persists it to
    `<trace-dir>/_meta.json` so `agentcap export` recovers it
    automatically. An explicit `filename=` opts back into
    overwrite-in-place — used only for "latest" pointer files.

14. **One output format only: parquet.** Single file via `--output`
    (local) or single file under a bucket prefix via `--push`. JSONL
    was dropped — it's a one-liner away from a parquet via
    `Dataset.from_parquet(...).to_json(...)`. Rendered token IDs
    (`rendered_tokens`) were dropped from rows for the same reason
    — deterministic from `(text, model)` and a 5-line recompute via
    `apply_chat_template`.

15. **Inference backend must deliver tool calls in `message.content`,
    not `message.reasoning_content`.** Hermes (and presumably other
    agents) parses tool calls from the OpenAI-spec `content` field.
    Reasoning-by-default models (Qwen 3.5+, etc.) on llama.cpp put
    their actual answer in `reasoning_content` and leave `content`
    empty — the agent loop sees no tool calls and stalls. Run
    llama-server with `--reasoning off` for these models;
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

4. **Per-agent compaction knob.** Tool-heavy multi-turn runs hit the
   model server's context limit (e.g. pi at ~70K tokens after one
   loop on a 64K-ctx server). Several agents expose native
   compaction — pi `/compact`, opencode `/compact`, goose
   `session-compact` — surface a `Driver.compact(session_id)`
   method and let the orchestrator call it when the previous turn's
   prompt-token count is approaching the configured ceiling.
