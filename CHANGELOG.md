# Changelog

All notable changes to agentcap. The project follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) loosely;
versions are not tagged yet — see `git log` for the authoritative
record.

## [Unreleased]

### Added

- Workspace model: `agentcap run` writes each invocation to
  `$AGENTCAP_WORKSPACE/.agentcap/<agent>-<provider>-<utc>/` (or
  `./.agentcap/...` if `AGENTCAP_WORKSPACE` is unset). `--workdir`
  remains as an explicit override.
- `agentcap ls` lists runs in the workspace with agent, model,
  task counts, and capture counts. `--long` adds upstream and
  full per-run detail.

### Changed

- `--push` now targets Hugging Face Dataset repos instead of
  Storage Buckets. Files land under `data/[<subdir>/]<file>.parquet`
  so the Hub Dataset Viewer renders them automatically. Repos are
  auto-created on first push, and a starter dataset card is seeded
  at the same time (left alone on subsequent pushes). One
  `agentcap export` invocation produces one git commit, regardless
  of how many runs it bundles. URI form switched from
  `hf://buckets/<owner>/<name>[/<subdir>]/` to
  `<owner>/<name>[/<subdir>]` (with optional `hf://datasets/` prefix).
  Consumers now load with `load_dataset("<owner>/<name>")`.
- `agentcap export` takes one or more run-ids (or workdir paths)
  positionally, with `--all` to push every run in the workspace.
  Agent is auto-read from each run's `run.json`; `--model` and
  `--agent` overrides are gone. `--push <dataset-repo>` is
  mandatory; `--output` removed.
- Internal "trace" terminology renamed to "capture" throughout
  (CLI: `--trace-dir` → `--capture-dir`; on-disk:
  `<workdir>/traces/` → `<workdir>/captures/`). "Trace" is reserved
  for the agent-native session files the community publishes under
  `format:agent-traces`.

### Removed

- `agentcap proxy` subcommand. Capturing arbitrary user traffic
  without a redaction/review layer (cf. pi-share-hf) raises
  security/ethical concerns that aren't in scope for agentcap. The
  proxy is now strictly an internal component of `agentcap run`.
- `agentcap.manifest` module and `build_manifest` function (earlier).
- `--workers` flag on `agentcap export`.

### Changed (earlier)

- Export is now a pure data shuffle. No tokenizer load, no
  chat-template render, no per-message token boundaries or role
  labels in the parquet. Token-level analysis is consumer-side
  via `transformers.AutoTokenizer.apply_chat_template`. The raw
  request and response bytes are preserved verbatim per row so
  consumers can re-render on their own terms.
- `_proxy.json` and the in-capture-dir metadata file dropped. The
  capture proxy now stamps `upstream_url` onto every
  `.request.json` and the export layer derives `provider` from
  that. Agent identity is supplied at export time via
  `--agent <name>` (or read from `<workdir>/run.json` by the
  example `export.sh` scripts).
- Default bucket filename now embeds `(agent, model, provider)`:
  `train-<agent>-<model>-<provider>-<utc>-<hex6>.parquet`.

### Security

- Pin `starlette>=1.0.1` to address GHSA-86qp-5c8j-p5mr (host
  header path poisoning in path-based middleware).

## Initial development

See `git log` for the full pre-1.0 history. Highlights:

- Capture proxy in front of `/v1/chat/completions` (any
  OpenAI-compat backend).
- Drivers for `hermes`, `goose`, `opencode`, `pi`.
- Per-agent sandbox (bwrap on Linux, Lima on macOS).
- HF Storage Bucket push (append-by-prefix, Xet-deduplicated).
- HF Router auth (auto-discovers `HF_TOKEN` /
  `~/.cache/huggingface/token`).
- Per-response upstream fingerprint (`Server`, `X-Served-By`,
  `X-Build-Info`, body-echoed `model`) for HF Router sub-provider
  routing visibility.
