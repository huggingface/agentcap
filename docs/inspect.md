# Inspecting captures

`agentcap inspect` is an fzf-driven picker over captured requests with
a live preview pane. The same picker chain runs over three sources:

| source                                  | how to pick it                                    |
|-----------------------------------------|---------------------------------------------------|
| local workspace (default)               | `agentcap inspect`                                |
| one local parquet file                  | `agentcap inspect --source path/to/file.parquet`  |
| HF dataset of captures                  | `agentcap inspect --source hf://datasets/<owner>/<name>` |

![inspect demo](img/inspect.gif)

_Regenerate from the repo root with `vhs docs/demo/inspect.tape` —
the tape's ``Output`` directive writes ``docs/img/inspect.gif``._

fzf is a hard requirement — install via `brew install fzf` or your
package manager. Without it, `inspect` errors out with a clear message
instead of falling back to a half-usable table dump.

## Workspace flow

Two-step pick: run → request → message.

```bash
agentcap inspect              # everything in the workspace
agentcap inspect <run-id>     # one run, skip the run picker
agentcap inspect <rid>        # bypass picker entirely, dump body
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

`--source <path>.parquet` skips the run picker and opens the request
picker directly against the parquet's rows. The preview pane reads
from the parquet (request body, diff vs the prior call, status,
size), and the message sub-picker drills into the synthesised
assistant reply too.

```bash
agentcap inspect --source ~/dev/runs/captures.parquet
```

If the parquet predates the `task_id` / `turn` schema (pre-2026-06-05
exports), the LOC column shows `-` and rows chain linearly per
`run_id` instead of per `(run, task)`. Otherwise LOC is
`task_01.1`, `task_01.2`, …, same as the workspace flow.

## HF dataset flow

Three-step pick: parquet → request → message.

```bash
agentcap inspect --source hf://datasets/dacorvo/hf-hub-session-captures
# or the bare shorthand:
agentcap inspect --source dacorvo/hf-hub-session-captures
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
# from the workspace
agentcap inspect 6437573b

# from a parquet
agentcap inspect 6437573b7a2d41abbb465a61a569351a \
    --source path/to/file.parquet

# from an HF dataset
agentcap inspect 6437573b7a2d41abbb465a61a569351a \
    --source hf://datasets/dacorvo/hf-hub-session-captures
```

Workspace lookups accept the 8-char prefix; parquet / HF lookups need
the full 32-char rid (exact match).
