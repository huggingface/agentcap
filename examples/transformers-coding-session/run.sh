#!/usr/bin/env bash
# Drive the transformers-coding-session corpus through any registered
# agent. The corpus is 30 open-ended planning/diagnostic prompts over
# huggingface/transformers — see tasks.txt.
#
# Prereqs:
#   1. An OpenAI-compat /v1 server on $UPSTREAM (default
#      http://127.0.0.1:8000). Easiest: llama.cpp with a GGUF, e.g.
#        GGUF_PATH=/path/to/model.gguf REASONING=off \
#            ./scripts/start_llama_cpp_server.sh
#   2. The agent binary on PATH (or wherever agentcap can find it).
#
# Usage:
#   ./run.sh --agent <name> --model <id>
#
# Examples:
#   ./run.sh --agent hermes --model Qwen/Qwen3-8B
#   ./run.sh --agent pi --model qwen3.6-35b-a3b
#   ./run.sh --agent goose --model gemma-4-26b
#
# ``--agent`` accepts any value listed by ``agentcap run --help``.
# ``--model`` is required for all agents.
#
# Captures + run.json land at
# ``$AGENTCAP_WORKSPACE/.agentcap/<agent>-<provider>-<utc>/`` (run
# ``agentcap ls`` to find the latest run). The export step at the end
# auto-selects the most recent one.
#
# Env knobs:
#   UPSTREAM        model server URL                http://127.0.0.1:8000
#   TURNS           multi-turn count                4
#   FOLLOWUP        continue | templates | synthesized   synthesized
#   TIMEOUT         per-turn timeout in seconds     300
#   TRANSFORMERS_CHECKOUT  path to a transformers git checkout. The
#                   script seeds <sandbox>/ as a detached
#                   ``git worktree`` of it so the agent has real
#                   transformers code to inspect — without this the
#                   sandbox is empty and the corpus prompts produce
#                   pure speculation.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
AGENT=""
MODEL=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --agent) AGENT="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        -h|--help)
            sed -n '/^# Usage:/,/^set -euo/p' "$0" | sed 's/^# \?//; /^set -euo/d'
            exit 0
            ;;
        *) echo "ERROR: unexpected arg: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "$AGENT" ]]; then
    echo "ERROR: --agent <name> is required. See: $0 --help" >&2
    exit 2
fi
if [[ -z "$MODEL" ]]; then
    echo "ERROR: --model <id> is required. See: $0 --help" >&2
    exit 2
fi

UPSTREAM="${UPSTREAM:-http://127.0.0.1:8000}"
TURNS="${TURNS:-4}"
FOLLOWUP="${FOLLOWUP:-synthesized}"
TIMEOUT="${TIMEOUT:-300}"

# Seed a sandbox dir with a transformers worktree so corpus prompts
# have real code to ground in. We materialise it once at $HERE/sandbox
# (idempotent across reruns) and pass it to agentcap via --sandbox.
if [[ -z "${TRANSFORMERS_CHECKOUT:-}" ]]; then
    for c in "$HOME/transformers" "$HERE/transformers"; do
        if [[ -d "$c/.git" || -f "$c/.git" ]]; then
            TRANSFORMERS_CHECKOUT="$(cd "$c" && pwd)"
            break
        fi
    done
fi
SANDBOX="$HERE/sandbox"
if [[ ! -e "$SANDBOX/.git" ]]; then
    if [[ -n "${TRANSFORMERS_CHECKOUT:-}" ]]; then
        echo "seeding $SANDBOX from transformers worktree at $TRANSFORMERS_CHECKOUT" >&2
        git -C "$TRANSFORMERS_CHECKOUT" worktree add --detach "$SANDBOX"
    else
        mkdir -p "$SANDBOX"
        echo "WARNING: no transformers checkout found; sandbox will be empty." >&2
        echo "         set TRANSFORMERS_CHECKOUT=<path> to seed it." >&2
    fi
fi

ARGS=(
    --agent     "$AGENT"
    --model     "$MODEL"
    --upstream  "$UPSTREAM"
    --sandbox   "$SANDBOX"
    --tasks     "$HERE/tasks.txt"
    --turns     "$TURNS"
    --followup  "$FOLLOWUP"
    --timeout   "$TIMEOUT"
)

agentcap run "${ARGS[@]}"

echo "done. captures land under \$AGENTCAP_WORKSPACE/.agentcap/ (run 'agentcap ls' to find this run)."
echo "next: $HERE/export.sh   # picks the most recent run and pushes"
