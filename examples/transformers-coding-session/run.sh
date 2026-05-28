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
#   ./run.sh --agent <name> --model <id> [WORKDIR]
#
# Examples:
#   ./run.sh --agent hermes --model Qwen/Qwen3-8B
#   ./run.sh --agent pi --model qwen3.6-35b-a3b
#   ./run.sh --agent goose --model gemma-4-26b
#
# ``--agent`` accepts any value listed by ``agentcap run --help``.
# ``--model`` is required for all agents.
#
# Env knobs:
#   UPSTREAM        model server URL                http://127.0.0.1:8000
#   LISTEN          in-process proxy bind HOST:PORT (default 127.0.0.1:8001)
#   TURNS           multi-turn count                4
#   FOLLOWUP        continue | templates | synthesized   synthesized
#   SYNTH_UPSTREAM  optional synth endpoint override (only when
#                   FOLLOWUP=synthesized). If unset, agentcap uses
#                   --upstream by default.
#   SYNTH_MODEL     optional synth model override (only when
#                   FOLLOWUP=synthesized). If unset, agentcap uses
#                   --model by default.
#   TIMEOUT         per-turn timeout in seconds     300
#   TRANSFORMERS_CHECKOUT  path to a transformers git checkout. The
#                   script seeds <WORKDIR>/sandbox as a detached
#                   ``git worktree`` of it so the agent has real
#                   transformers code to inspect — without this the
#                   sandbox is empty and the corpus prompts produce
#                   pure speculation.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
AGENT=""
MODEL=""
WORKDIR=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --agent) AGENT="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        -h|--help)
            sed -n '/^# Usage:/,/^set -euo/p' "$0" | sed 's/^# \?//; /^set -euo/d'
            exit 0
            ;;
        *) WORKDIR="$1"; shift ;;
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

WORKDIR="${WORKDIR:-$HERE/runs/$AGENT-$(date +%Y-%m-%d-%H%M)}"
UPSTREAM="${UPSTREAM:-http://127.0.0.1:8000}"
TURNS="${TURNS:-4}"
FOLLOWUP="${FOLLOWUP:-synthesized}"
SYNTH_UPSTREAM="${SYNTH_UPSTREAM:-}"
SYNTH_MODEL="${SYNTH_MODEL:-}"
TIMEOUT="${TIMEOUT:-300}"

# Seed <WORKDIR>/sandbox with a transformers worktree so the corpus
# prompts have real code to ground in. Skip if sandbox already
# contains a .git (idempotent across reruns).
if [[ -z "${TRANSFORMERS_CHECKOUT:-}" ]]; then
    for c in "$HOME/transformers" "$HERE/transformers"; do
        if [[ -d "$c/.git" || -f "$c/.git" ]]; then
            TRANSFORMERS_CHECKOUT="$(cd "$c" && pwd)"
            break
        fi
    done
fi
mkdir -p "$WORKDIR"
SANDBOX="$WORKDIR/sandbox"
if [[ ! -e "$SANDBOX/.git" ]]; then
    if [[ -n "${TRANSFORMERS_CHECKOUT:-}" ]]; then
        echo "seeding $SANDBOX from transformers worktree at $TRANSFORMERS_CHECKOUT" >&2
        git -C "$TRANSFORMERS_CHECKOUT" worktree add --detach "$SANDBOX"
    else
        echo "WARNING: no transformers checkout found; sandbox will be empty." >&2
        echo "         set TRANSFORMERS_CHECKOUT=<path> to seed it." >&2
    fi
fi

ARGS=(
    --agent     "$AGENT"
    --upstream  "$UPSTREAM"
    --workspace "$SANDBOX"
    --tasks     "$HERE/tasks.txt"
    --turns     "$TURNS"
    --followup  "$FOLLOWUP"
    --workdir   "$WORKDIR"
    --timeout   "$TIMEOUT"
)
[[ -n "${LISTEN:-}" ]] && ARGS+=(--listen "$LISTEN")
[[ -n "$MODEL" ]] && ARGS+=(--model "$MODEL")
if [[ "$FOLLOWUP" == "synthesized" ]]; then
    [[ -n "${SYNTH_UPSTREAM:-}" ]] && ARGS+=(--synth-upstream "$SYNTH_UPSTREAM")
    [[ -n "${SYNTH_MODEL:-}" ]] && ARGS+=(--synth-model "$SYNTH_MODEL")
fi

agentcap run "${ARGS[@]}"

echo "done. captures in $WORKDIR/captures, summary in $WORKDIR/run.json"
echo "next: $HERE/export.sh \"$WORKDIR\"   # render + push to the corpus bucket"
