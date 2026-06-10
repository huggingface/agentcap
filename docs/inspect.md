# Inspecting captures

`agentcap inspect` is an fzf-driven picker over captured requests with
a live preview pane. A single positional `TARGET` is classified by
content — does the path exist? does it look like an hf URI? all hex?
— and routed accordingly. `inspect` does NOT consult
`$AGENTCAP_WORKSPACE`; what you point it at is what you get.

| target shape                                | what it picks                |
|---------------------------------------------|------------------------------|
| (omitted)                                   | cwd's `.agentcap/`           |
| `<dir>` (exists locally)                    | that dir's `.agentcap/`      |
| `<run-id>` (under cwd's `.agentcap/`)       | scoped to that run           |
| `<rid>` (32 hex)                            | body dump (cwd workspace)    |
| `<file>.parquet`                            | parquet picker               |
| `hf://datasets/<owner>/<name>`              | HF dataset picker            |
| `<owner>/<name>` (not a local dir)          | HF dataset (shorthand)       |

![inspect demo](img/inspect.gif)

_Regenerate from the repo root with `vhs docs/demo/inspect.tape` —
the tape's ``Output`` directive writes ``docs/img/inspect.gif``._

fzf is a hard requirement — install via `brew install fzf` or your
package manager. Without it, `inspect` errors out with a clear message
instead of falling back to a half-usable table dump.

## Workspace flow

Two-step pick: run → request → message.

```bash
agentcap inspect                          # cwd workspace
agentcap inspect <dir>                    # another local workspace
agentcap inspect <run-id>                 # one run, skip the run picker
agentcap inspect <rid>                    # bypass picker, dump body
```

In the picker:

- **Type** to fuzzy-filter. Matches against every visible column
  *and* against the full new-message content of each row (so a query
  like `hf-cli` matches rows where the term appears 4 messages back
  in the diff, not just on the visible row).
- **Enter** drills into the picked row — first into the request's
  flattened conversation (one row per message + tool call + decoded
  response), then on a second Enter, dumps that message's content.
- **Esc** walks back one level: message → request → run → exit.

fzf operators work in the search bar: `'word` for exact match,
`^word` / `word$` for anchors, `!word` to exclude, `|` to OR.
Each non-negated term is highlighted in red inside the preview pane.

## Parquet flow

Passing a `.parquet` path skips the run picker and opens the request
picker directly against the parquet's rows. The preview pane reads
from the parquet (request body, diff vs the prior call, status,
size), and the message sub-picker drills into the synthesised
assistant reply too.

```bash
agentcap inspect ~/dev/runs/captures.parquet
```

If the parquet predates the `task_id` / `turn` schema (pre-2026-06-05
exports), the LOC column shows `-` and rows chain linearly per
`run_id` instead of per `(run, task)`. Otherwise LOC is
`task_01.1`, `task_01.2`, …, same as the workspace flow.

## HF dataset flow

Three-step pick: parquet → request → message.

```bash
agentcap inspect hf://datasets/dacorvo/hf-hub-session-captures
# or the bare shorthand (only recognised as HF when no local dir
# by that name exists):
agentcap inspect dacorvo/hf-hub-session-captures
```

Level 1 (parquet picker) shows one row per `.parquet` file in the
dataset with `AGENT  MODEL  ROWS  SIZE  PATH`. The preview pane shows
row count, model id, and a sorted task list with the first user
prompt + completed-turn count per task (mirrors the local
`_run_preview` shape).

Picked parquets are downloaded once into the `huggingface_hub` cache
(`~/.cache/huggingface/hub/`) — subsequent picks of the same file are
instant.

### Caching

Level-1 metadata (model, row count, task list) is also cached on disk
under `~/.cache/agentcap/hf-list/<repo>/<file>.json`, keyed by the
parquet's git `blob_id`. Net effect:

- First open of a fresh dataset: ~10s for ~20 parquets (parallel
  prefetch, fanout × 8).
- Subsequent opens + every fzf hover: instant (cache hit).
- If a file at the same path gets re-uploaded with different content,
  the blob_id changes and the cache entry is automatically refetched.

## Piping the picked rid

`--rid` makes the picker print the selected request id and exit
instead of opening it — useful for chaining into `agentcap replay`.

```bash
agentcap replay $(agentcap inspect --rid) --target http://127.0.0.1:8000
```

## Looking up a specific rid

When you already know the rid, skip the picker:

```bash
# rid lookup is against cwd's ``.agentcap/`` workspace (8-char
# prefix accepted; conflicts surface as an "ambiguous" error).
agentcap inspect 6437573b
```

To dump a specific rid from a parquet or an HF dataset, open the
picker with that source and pick the rid:

```bash
agentcap inspect path/to/file.parquet
agentcap inspect hf://datasets/dacorvo/hf-hub-session-captures
```
