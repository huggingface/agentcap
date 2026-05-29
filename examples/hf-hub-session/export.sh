#!/usr/bin/env bash
# Render captures from this corpus into parquet and push to a Hugging
# Face Dataset repo under the ``hf-hub-session/`` subdir.
#
# Pins AGENTCAP_WORKSPACE to the corpus dir so ``agentcap export``
# only sees runs from this corpus (run.sh does the same).
#
# Usage:
#   ./export.sh                       # latest run
#   ./export.sh <run-id> [<run-id>…]  # explicit run-ids (see `agentcap ls`)
#   ./export.sh --all                 # every run in $HERE/.agentcap/
#
# Env knobs:
#   DATASET   default --push target. dacorvo/agentcap-captures/hf-hub-session
#   AGENTCAP  path to the agentcap binary; default: `agentcap` on PATH

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
export AGENTCAP_WORKSPACE="$HERE"

DEFAULT_DATASET="dacorvo/agentcap-captures/hf-hub-session"
AGENTCAP="${AGENTCAP:-agentcap}"
PUSH="${DATASET:-$DEFAULT_DATASET}"

if [[ "${1:-}" == "--all" ]]; then
    exec "$AGENTCAP" export --all --push "$PUSH"
fi
if [[ $# -gt 0 ]]; then
    exec "$AGENTCAP" export "$@" --push "$PUSH"
fi

LATEST="$(ls -td "$HERE"/.agentcap/*/ 2>/dev/null | head -1 | sed 's:/$::')"
if [[ -z "$LATEST" ]]; then
    echo "ERROR: no runs under $HERE/.agentcap/." >&2
    exit 2
fi
echo "auto-selected latest run: $(basename "$LATEST")" >&2
exec "$AGENTCAP" export "$(basename "$LATEST")" --push "$PUSH"
