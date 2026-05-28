# Roadmap

The producer side (runner + capture proxy + parquet export + bucket
push) is in place. The one missing piece is the consumer side: a way
to look at the captured parquets without writing a notebook.

## Inspector Space

Public Gradio CPU Space, pure parquet reader. Loads via
`load_dataset("hf://buckets/.../<prefix>/")`. Three views:

1. **Session timeline** — chat-style thread, system prompt collapsed,
   tool calls as expandable cards showing `(tool_name, arguments,
   result)`. The front door.
2. **Rendered-bytes pane** — re-render `request.messages` through
   the model's chat template (consumer-side via
   `apply_chat_template`) and colour-code per role. Producer ships
   the bytes; the Space owns the render.
3. **Cross-cell comparison** — pick the same task across two
   `(agent, model)` parquets, see how the tool-call pattern, turn
   count, and token spend differ.

Stretch:
- Stable-prefix highlight on the rendered-bytes pane — ties directly
  to kv-cache-reuse recurrence research.
- Filter: "all sessions where tool X was called with args matching Y".

## Out of scope

- **Replay.** Re-issuing captured requests against a different model
  is per-request only — the agent's next prompt diverges the moment
  the model responds differently. The only useful per-request replay
  is the splice-correctness harness used by kv-cache-reuse research,
  which doesn't need agentcap support.
