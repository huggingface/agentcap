# Roadmap

The producer side (runner + capture proxy + parquet export + dataset
push) is in place. Pushed datasets get the Hub Dataset Viewer for
free; a custom inspector would only earn its maintenance if it added
something the viewer can't.

## Inspector Space (probably not needed)

The free Hub Dataset Viewer covers the common case (paginated table,
filter, row expand). A custom Gradio Space would only earn its
maintenance if it added something the viewer can't — chat-style
timeline rendering, cross-`(agent, model)` comparison on the same
task, or kv-cache-reuse prefix-stability highlighting. Three views
the Space could host:

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

## Single-turn replay (shipped)

`agentcap replay <rid> --target <url>` resolves a captured request
by id and re-POSTs the body verbatim to an OpenAI-compatible
endpoint. Workspace-relative by default; `--source` accepts a
capture dir, a `.parquet`, or `hf://datasets/<owner>/<name>` for
re-issuing requests from a published dataset. `agentcap inspect
<rid>` prints the body to stdout for piping into `curl` / stashing
as a regression fixture. The library reuse surface is
`agentcap.replay.{load_request, load_requests}`.

## Out of scope

- **Multi-turn (conversation) replay.** Diverges the moment the
  model responds differently — the agent's next prompt depends on
  the response. Only per-request replay (above) is meaningful, and
  that ships.
