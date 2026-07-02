#!/usr/bin/env bash
# Drive a coding agent through a recall task inside an agentcap sandbox, with
# funes wired in as a first-class `recall` tool. The `funes` binary + a prewarmed
# model cache come from the bundle mounted via `agentcap run --tool-dir` (build it
# once with ./build-funesenv.sh); the index is the live hf:// shared store, read
# over the sandbox's network.
#
# The bundle's `tool_init.sh` (run in the sandbox by agentcap, with $AGENTCAP_AGENT set)
# does the per-agent wiring, so this script is agent-agnostic:
#   pi        no MCP client → `funes install pi` seeds funes' bridge extension into the
#             cwd's .pi/extensions/; pi auto-discovers it.
#   hermes    native MCP client → register funes as the MCP server `funes` (mcp_funes_*).
#             Note: hermes' built-in `session_search` is tried first, so funes is a fallback.
#   opencode  native MCP client → drop a cwd opencode.json registering funes (funes_recall).
#
# Usage:
#   ./run.sh --model <id> [--agent pi|hermes|opencode] [--tasks <file>]
#
# Example:
#   ./run.sh --model GLM-4.5-Air                  # pi (default)
#   ./run.sh --model GLM-4.5-Air --agent hermes
#   ./run.sh --model GLM-4.5-Air --agent opencode
#
# Captures land under $HERE/.agentcap/<run-id>/; publish with `agentcap export`.
#
# Env knobs:
#   UPSTREAM   model server URL                http://127.0.0.1:8001
#   TURNS      turns per task                  1
#   TIMEOUT    per-turn timeout (seconds)      900
#   HF_TOKEN   token for a private FUNES_REMOTE (else ~/.cache/huggingface/token)

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
export AGENTCAP_WORKSPACE="$HERE"

MODEL="" TASKS="$HERE/tasks.txt" AGENT="pi"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --agent) AGENT="$2"; shift 2 ;;
        --tasks) TASKS="$2"; shift 2 ;;
        -h|--help) sed -n '/^# Usage:/,/^set -euo/p' "$0" | sed 's/^# \?//; /^set -euo/d'; exit 0 ;;
        *) echo "ERROR: unexpected arg: $1" >&2; exit 2 ;;
    esac
done
[[ -n "$MODEL" ]] || { echo "ERROR: --model is required. See: $0 --help" >&2; exit 2; }
[[ "$AGENT" == pi || "$AGENT" == hermes || "$AGENT" == opencode ]] || { echo "ERROR: --agent must be pi, hermes, or opencode" >&2; exit 2; }
[[ -f "$TASKS" ]] || { echo "ERROR: tasks file not found: $TASKS" >&2; exit 2; }

UPSTREAM="${UPSTREAM:-http://127.0.0.1:8001}"
TURNS="${TURNS:-1}"
TIMEOUT="${TIMEOUT:-900}"

BUNDLE="$HERE/funesenv"
[[ -x "$BUNDLE/bin/funes" ]] || {
    echo "ERROR: funes bundle missing at $BUNDLE. Build it first:" >&2
    echo "         FUNES_REMOTE=<org>/<repo> ./build-funesenv.sh" >&2
    exit 2
}

# The default index is public. For a private FUNES_REMOTE, agentcap doesn't forward host
# env into the sandbox, so if an HF token is available, drop it at the standard
# huggingface_hub path inside the bundle; the wrapper reads it into HF_TOKEN. It's
# gitignored, and a read-only dataset-scoped token is plenty.
TOKEN="${HF_TOKEN:-}"
[[ -z "$TOKEN" && -f "$HOME/.cache/huggingface/token" ]] && TOKEN="$(cat "$HOME/.cache/huggingface/token")"
if [[ -n "$TOKEN" ]]; then
    TOKEN_FILE="$BUNDLE/funes-home/.cache/huggingface/token"
    mkdir -p "$(dirname "$TOKEN_FILE")"
    install -m 600 /dev/null "$TOKEN_FILE"
    printf '%s' "$TOKEN" > "$TOKEN_FILE"
fi

# Fresh, otherwise-empty sandbox cwd each run — nothing to grep, so the only path
# to the answer is recall. Per-agent funes wiring (pi extension / hermes MCP) is done
# by the bundle's tool_init.sh, which agentcap runs in the sandbox; nothing to seed here.
SANDBOX="$HERE/sandbox"
rm -rf "$SANDBOX"
mkdir -p "$SANDBOX"

echo ">>> agent=$AGENT model=$MODEL tasks=$(basename "$TASKS") upstream=$UPSTREAM" >&2
agentcap run \
    --agent     "$AGENT" \
    --model     "$MODEL" \
    --upstream  "$UPSTREAM" \
    --sandbox   "$SANDBOX" \
    --tool-dir  "$BUNDLE" \
    --label     "funes-recall" \
    --tasks     "$TASKS" \
    --turns     "$TURNS" \
    --timeout   "$TIMEOUT"

echo "done. captures under $HERE/.agentcap/ (agentcap ls). publish: agentcap export" >&2
