#!/usr/bin/env bash
# Drive the hf-hub-session corpus through any registered agent. The
# corpus is 30 open-ended exploration prompts over the Hugging Face
# Hub (models / datasets / spaces / papers); the agent is expected to
# reach for the ``hf`` CLI, the ``huggingface_hub`` SDK, or direct
# HTTPS calls to https://huggingface.co/api/...  as its primary tool
# surface. The sandbox carries the huggingface/skills bundle so the
# agent can ``skill_view(hf-cli)`` etc.; without it the loop ends in
# "skill not found" with no grounding.
#
# Prereqs:
#   1. An OpenAI-compat /v1 server on $UPSTREAM (default
#      http://127.0.0.1:8000). Easiest: llama.cpp + GGUF.
#   2. The agent binary on PATH (or wherever agentcap can find it).
#   3. Hub credentials available to the agent if it needs private
#      repos (e.g. via $HF_TOKEN or `hf auth login`).
#   4. A local clone of huggingface/skills. The script auto-detects
#      $HOME/skills, $HOME/dev/skills, or $HERE/skills; if none of
#      these exists, it clones into $HOME/.cache/agentcap/hf-skills
#      on first run. Override with SKILLS_CHECKOUT=<path>.
#
# Usage:
#   ./run.sh --agent <name> --model <id>
#
# Examples:
#   ./run.sh --agent hermes --model Qwen/Qwen3-8B
#   ./run.sh --agent goose --model gemma-4-26B-A4B-it
#
# Captures + run.json land under ``$HERE/.agentcap/<run-id>/``.
#
# Env knobs:
#   UPSTREAM        model server URL                http://127.0.0.1:8000
#   TURNS           multi-turn count                4
#   FOLLOWUP        continue | templates | synthesized   synthesized
#   TIMEOUT         per-turn timeout in seconds     300
#   SKILLS_CHECKOUT path to a huggingface/skills clone (see Prereqs.4)

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
export AGENTCAP_WORKSPACE="$HERE"

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

if [[ -z "$AGENT" || -z "$MODEL" ]]; then
    echo "ERROR: --agent and --model are required. See: $0 --help" >&2
    exit 2
fi

UPSTREAM="${UPSTREAM:-http://127.0.0.1:8000}"
TURNS="${TURNS:-4}"
FOLLOWUP="${FOLLOWUP:-synthesized}"
TIMEOUT="${TIMEOUT:-300}"

# Resolve a huggingface/skills checkout; missing skills is fatal for
# this corpus (agents spin in "skill not found" with no fallback).
if [[ -z "${SKILLS_CHECKOUT:-}" ]]; then
    for c in "$HOME/skills" "$HOME/dev/skills" "$HERE/skills"; do
        if [[ -d "$c/skills" && -f "$c/agents/AGENTS.md" ]]; then
            SKILLS_CHECKOUT="$(cd "$c" && pwd)"
            break
        fi
    done
fi
if [[ -z "${SKILLS_CHECKOUT:-}" ]]; then
    SKILLS_CHECKOUT="$HOME/.cache/agentcap/hf-skills"
    if [[ ! -d "$SKILLS_CHECKOUT/skills" || ! -f "$SKILLS_CHECKOUT/agents/AGENTS.md" ]]; then
        echo "[skills] no local huggingface/skills clone found; fetching into $SKILLS_CHECKOUT" >&2
        mkdir -p "$(dirname "$SKILLS_CHECKOUT")"
        rm -rf "$SKILLS_CHECKOUT"
        git clone --depth 1 https://github.com/huggingface/skills "$SKILLS_CHECKOUT"
    fi
fi
if [[ ! -d "$SKILLS_CHECKOUT/skills" || ! -f "$SKILLS_CHECKOUT/agents/AGENTS.md" ]]; then
    echo "ERROR: SKILLS_CHECKOUT=$SKILLS_CHECKOUT does not look like a huggingface/skills clone." >&2
    echo "       Expected skills/ and agents/AGENTS.md under it." >&2
    exit 2
fi
echo "[skills] using checkout: $SKILLS_CHECKOUT" >&2

SANDBOX="$HERE/sandbox"
mkdir -p "$SANDBOX"

agentcap run \
    --agent     "$AGENT" \
    --model     "$MODEL" \
    --upstream  "$UPSTREAM" \
    --sandbox   "$SANDBOX" \
    --skills    "$SKILLS_CHECKOUT" \
    --tasks     "$HERE/tasks.txt" \
    --turns     "$TURNS" \
    --followup  "$FOLLOWUP" \
    --timeout   "$TIMEOUT"

echo "done. captures land under $HERE/.agentcap/ (run 'agentcap ls' to list)."
echo "next: $HERE/export.sh   # picks the most recent run and pushes"
