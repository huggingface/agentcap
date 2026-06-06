# Replaying a captured request

`agentcap replay` re-issues a captured request body against any
OpenAI-compatible endpoint. The captured JSON is sent verbatim — same
messages, same tools, same temperature — so any difference in output
comes from the **target** model, not from agentcap normalising the
payload.

> **Demo** — _GIF placeholder. Regenerate from the repo root with
> `vhs docs/demo/replay.tape` (the tape's ``Output`` directive
> writes ``docs/img/replay.gif``)._
>
> `![replay](img/replay.gif)`

## Minimum command

```bash
agentcap replay <rid> --target http://127.0.0.1:8000
```

- `<rid>` — full or 8-char-prefix request id from a captured run.
- `--target` — OpenAI-compat endpoint root (no `/v1` suffix).

The captured body is POSTed to `<target>/v1/chat/completions` and the
target's response streams to stdout.

## Picking the rid interactively

Without a request id, `replay` opens the same picker chain as
`inspect`. Pick a request, hit Enter, the body is sent.

```bash
agentcap replay --target http://127.0.0.1:8000
```

The `--rid` / picker / source plumbing is shared with `inspect`, so
the same flags work:

```bash
# Replay against a different target with a body sourced from a parquet
agentcap replay --target https://router.huggingface.co \
    --source path/to/captures.parquet

# Or from an HF dataset
agentcap replay --target https://router.huggingface.co \
    --source hf://datasets/dacorvo/hf-hub-session-captures
```

## What lands on stdout

By default, replay streams the rendered generation:

- Assistant text content as it arrives.
- Tool calls rendered inline as `[tool:NAME](args)` markers when the
  target emits them.

For the raw upstream bytes (the actual SSE / JSON payload from the
target, not agentcap's rendering), pass `--raw`:

```bash
agentcap replay <rid> --target http://127.0.0.1:8000 --raw
```

Status (HTTP code, model echoed back, byte count, duration) goes to
stderr in both modes.

## Single-turn only

`replay` re-issues exactly one captured turn. Multi-turn replay is
intentionally out of scope: as soon as the new model responds
differently from the captured one, the conversation diverges and the
subsequent captured turns don't apply.

## Common uses

- **Bisecting model regressions** — replay the same captured request
  against `model-a` and `model-b`, diff the outputs.
- **A/B-ing a chat template change** — same body, different upstream
  serving the same model with a different template / quant.
- **Reproducing a tool-call failure** — capture once, then replay
  while iterating on tool definitions or system prompt.

## What's NOT preserved

The captured body is parsed JSON, not the original byte sequence. So
the dict shape is identical but key ordering may differ. If your
target's hashing / content addressing depends on byte-exact input,
replay won't reproduce the original byte stream — only the semantic
payload.
