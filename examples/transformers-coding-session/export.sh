#!/usr/bin/env bash
# Render the captures from a `run.sh` workdir into parquet and push to
# the agentcap-captures Dataset repo under the
# `transformers-coding-session/` subdir.
#
# Prereqs:
#   1. A workdir from run.sh (or pass --all to push every workdir under
#      $HERE/runs/).
#   2. `hf auth login` (read+write) for the target dataset.
#
# Usage:
#   ./export.sh [WORKDIR] [--push <repo>]   # one run (defaults to latest)
#   ./export.sh --all [--push <repo>]       # every run under runs/
#
# Env knobs:
#   DATASET   default --push target.
#             dacorvo/agentcap-captures/transformers-coding-session
#   AGENTCAP  path to the agentcap binary; default: `agentcap` on PATH

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_DATASET="dacorvo/agentcap-captures/transformers-coding-session"
AGENTCAP="${AGENTCAP:-agentcap}"
DATASET="${DATASET:-$DEFAULT_DATASET}"

WORKDIR=""
ALL=""
PUSH=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --all)    ALL="1"; shift ;;
        --push)   PUSH="$2"; shift 2 ;;
        -h|--help)
            sed -n '/^# Usage:/,/^set -euo/p' "$0" | sed 's/^# \?//; /^set -euo/d'
            exit 0
            ;;
        *) WORKDIR="$1"; shift ;;
    esac
done

PUSH="${PUSH:-$DATASET}"

if [[ -n "$ALL" ]]; then
    if [[ ! -d "$HERE/runs" ]]; then
        echo "ERROR: $HERE/runs is empty." >&2
        exit 2
    fi
    ARGS=()
    for d in "$HERE"/runs/*/; do
        ARGS+=("${d%/}")
    done
    exec "$AGENTCAP" export "${ARGS[@]}" --push "$PUSH"
fi

if [[ -z "$WORKDIR" ]]; then
    WORKDIR="$(ls -td "$HERE"/runs/*/ 2>/dev/null | head -1 | sed 's:/$::')"
    if [[ -z "$WORKDIR" ]]; then
        echo "ERROR: no WORKDIR given and $HERE/runs is empty." >&2
        exit 2
    fi
    echo "auto-selected latest workdir: $WORKDIR" >&2
fi

exec "$AGENTCAP" export "$WORKDIR" --push "$PUSH"
