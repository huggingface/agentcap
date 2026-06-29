# Exporting to a Hugging Face dataset

`agentcap export` bundles one or more runs into parquet, ships each
agent's native traces alongside, scans for secrets (offline, never
verifying), and pushes everything as a [HF Collection][coll].

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
directory **offline** and aborts on **any** pattern hit. It never
verifies: verifying round-trips the live credential to the provider,
whose secret-scanning then revokes it (for an HF token, verifying *is*
the revocation). The cost is that benign high-entropy strings can also
trip the gate — inspect them (`agentcap inspect`), then redact or
`--no-scan`.

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

One row per captured chat-completion request. Columns:

| column              | source                                  |
|---|---|
| `request_id`        | proxy-minted UUID                       |
| `model`             | `request.body.model`                    |
| `captured_at`       | request capture epoch                   |
| `task_id`           | orchestrator-side metadata — which corpus task this call belongs to |
| `turn`              | orchestrator-side metadata — which turn of the task |
| `request`           | JSON-stringified raw OpenAI request     |
| `response`          | JSON-stringified raw response (or `{stream: true, raw: "<SSE bytes>"}` for streamed) |
| `served_by`         | per-response `X-Served-By` header (HF Router sub-provider routing) |
| `served_build_info` | per-response `X-Build-Info` header      |
| `served_model`      | per-response body-echoed `model`        |
| `provider`          | derived from `upstream_url` hostname (constant per file) |
| `upstream_url`      | proxy upstream at capture time (constant per file) |

The `request` and `response` columns are JSON strings (not nested
structs) so Arrow doesn't infer a schema over heterogeneous tool-call
fields. Consumers `json.loads` them. `task_id` / `turn` let
downstream picker UIs group rows by task without parsing the
message history; older parquets without those columns keep working
(consumers treat missing values as unknown).

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
agentcap inspect hf://datasets/my-org/my-captures-captures
```

See [docs/inspect.md](inspect.md) for the picker UX.
