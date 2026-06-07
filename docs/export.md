# Exporting to a Hugging Face dataset

`agentcap export` bundles one or more runs into parquet, ships each
agent's native traces alongside, scans for verified secrets, and
pushes everything as a [HF Collection][coll].

[coll]: https://huggingface.co/docs/hub/collections

![export demo](img/export.gif)

_Regenerate from the repo root with `vhs docs/demo/export.tape` —
the tape's ``Output`` directive writes ``docs/img/export.gif``._

## Minimum command

```bash
agentcap export --all --push my-org/my-captures
```

One `--push <owner>/<base>` value drives three artefacts:

| repo                            | content                                                       |
|---------------------------------|---------------------------------------------------------------|
| `<owner>/<base>-captures`       | one parquet per run, one row per chat-completion call         |
| `<owner>/<base>-<agent>-traces` | the agent's native session log files (one repo per agent)     |
| Collection `<base>` under `<owner>` | groups the above so they surface together on the Hub        |

The collection + dataset cards are created on first push; subsequent
pushes leave any existing card untouched.

## Selecting what to export

```bash
# Every run in the workspace.
agentcap export --all --push my-org/my-captures

# Specific runs (run-ids from `agentcap ls`).
agentcap export hermes-local-20260512-162345 \
    goose-local-20260512-170000 \
    --push my-org/my-captures

# Arbitrary workdir / capture dir.
agentcap export ./some/workdir --push my-org/my-captures
```

## Secret scanning

Before pushing, `agentcap export` runs `trufflehog` against each run
directory and aborts on any **verified** secret hit (pattern-only
matches are surfaced but don't block).

```bash
brew install trufflehog        # macOS
# or
sudo apt install -y trufflehog # Debian/Ubuntu
```

Pass `--no-scan` to skip the gate — only do this for runs you're sure
contain no real credentials.

## Filename layout

Each run's captures parquet lands at
`data/train-<agent>-<model>-<provider>-<utc>-<hex>.parquet`. The UTC
timestamp + hex hash are stamped at **export** time (not capture
time), so re-exporting the same run produces a fresh filename
alongside the old one. If you want only the new version, delete the
old via `huggingface_hub.HfApi().delete_file(...)` after the push.

## Parquet schema

One row per captured chat-completion request. See the
[README's "Parquet schema" section][schema] for the full column list;
the orchestrator-side `task_id` + `turn` columns (added 2026-06-05)
let downstream picker UIs group rows by task without parsing the
message history.

[schema]: ../README.md#parquet-schema

## Traces layout

Per-agent traces land in their own repo because each agent's native
log format is different (opencode SQLite, pi JSONL, hermes SQLite,
goose log file). Splitting per agent means a downstream consumer
doesn't have to filter by agent before unpacking files of mixed
formats.

```
my-org/my-captures-hermes-traces/
└── traces/<run-id>/
    ├── task_01-session.sqlite
    ├── task_02-session.sqlite
    └── …
```

A traces repo is created only for agents that actually produced
traces in the exported runs.

## Inspecting after export

The exported parquet is fully browsable via `agentcap inspect`:

```bash
agentcap inspect --source hf://datasets/my-org/my-captures-captures
```

See [docs/inspect.md](inspect.md) for the picker UX.
