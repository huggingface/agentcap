# Changelog

All notable changes to agentcap. The project follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) loosely;
versions are not tagged yet — see `git log` for the authoritative
record.

## [Unreleased]

### Changed

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

### Removed

- `agentcap.manifest` module and `build_manifest` function.
- `--workers` flag on `agentcap export` (no parallel render to
  drive any more).

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
