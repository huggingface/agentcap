#!/usr/bin/env bash
# Render the traces from a `run.sh` workdir into parquet and (by
# default) push to the agentcap-traces Storage Bucket. Wraps `agentcap
# export` with the conventions this corpus has settled on:
#
#   - bucket prefix per corpus (`transformers-coding-session/`)
#   - default filename embeds (agent, model) so a single prefix can
#     hold many tuples side by side: render lands at
#     <prefix>/train-<agent>-<model>-<ts>-<hex>.parquet
#
# Prereqs:
#   1. A trace dir from `run.sh` (i.e. <WORKDIR>/traces with one
#      *.request.json + *.response.json pair per captured request, and
#      ideally <WORKDIR>/traces/_meta.json so the agent name is
#      auto-detected).
#   2. No GPU-bound model server active on the same host. transformers'
#      CUDA init during tokenizer/processor load crashes a co-running
#      llama-server. Tear down the capture run before exporting.
#   3. `hf auth login` (read+write) for the target bucket.
#
# Usage:
#   ./export.sh [WORKDIR] [--model <id>] [--output <path> | --push <uri>]
#
# Examples:
#   # Push the latest run's traces to the default corpus bucket:
#   ./export.sh
#
#   # Specific workdir, explicit HF model id (override or fill-in when
#   # the captured `model` field is the llama-server alias rather than
#   # the HF repo id):
#   ./export.sh runs/goose-2026-05-07-1248 --model google/gemma-4-E4B-it
#
#   # Local parquet instead of bucket push:
#   ./export.sh --output /tmp/run.parquet
#
# Env knobs:
#   BUCKET         default `--push` URI when --push/--output not given.
#                  hf://buckets/dacorvo/agentcap-traces/transformers-coding-session/
#   WORKERS        parallel render workers (default 8).
#   AGENTCAP       path to the agentcap binary; default: `agentcap` on PATH

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_BUCKET="hf://buckets/dacorvo/agentcap-traces/transformers-coding-session/"
AGENTCAP="${AGENTCAP:-agentcap}"
BUCKET="${BUCKET:-$DEFAULT_BUCKET}"
WORKERS="${WORKERS:-8}"

WORKDIR=""
MODEL=""
OUTPUT=""
PUSH=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)  MODEL="$2"; shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        --push)   PUSH="$2"; shift 2 ;;
        -h|--help)
            sed -n '/^# Usage:/,/^set -euo/p' "$0" | sed 's/^# \?//; /^set -euo/d'
            exit 0
            ;;
        *) WORKDIR="$1"; shift ;;
    esac
done

# Default to the most-recent run dir under ./runs/.
if [[ -z "$WORKDIR" ]]; then
    if [[ -d "$HERE/runs" ]]; then
        WORKDIR="$(ls -td "$HERE"/runs/*/ 2>/dev/null | head -1 | sed 's:/$::')"
    fi
    if [[ -z "$WORKDIR" ]]; then
        echo "ERROR: no WORKDIR given and $HERE/runs is empty." >&2
        echo "       Run ./run.sh first, or pass a path explicitly." >&2
        exit 2
    fi
    echo "auto-selected latest workdir: $WORKDIR" >&2
fi

TRACES="$WORKDIR/traces"
if [[ ! -d "$TRACES" ]]; then
    echo "ERROR: $TRACES is not a directory." >&2
    exit 2
fi

if [[ -z "$OUTPUT" && -z "$PUSH" ]]; then
    PUSH="$BUCKET"
fi

ARGS=("$TRACES" --workers "$WORKERS")
[[ -n "$MODEL"  ]] && ARGS+=(--model  "$MODEL")
[[ -n "$OUTPUT" ]] && ARGS+=(--output "$OUTPUT")
[[ -n "$PUSH"   ]] && ARGS+=(--push   "$PUSH")

echo "$AGENTCAP export ${ARGS[*]}" >&2
exec "$AGENTCAP" export "${ARGS[@]}"
