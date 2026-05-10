#!/usr/bin/env bash
# Drive the hf-hub-session corpus through any registered agent. The
# corpus is 30 open-ended exploration prompts over the Hugging Face
# Hub (models / datasets / spaces / papers) — the agent is expected
# to reach for the `hf` CLI, the `huggingface_hub` SDK, or direct
# HTTPS calls to https://huggingface.co/api/... as its primary tool
# surface. The sandbox is intentionally empty: there is no local
# repo to grep; the agent must ground every answer in the live Hub.
#
# Prereqs:
#   1. An OpenAI-compat /v1 server on $UPSTREAM (default
#      http://127.0.0.1:8000). Easiest: llama.cpp + GGUF.
#   2. The agent binary on PATH (or wherever agentcap can find it).
#   3. Hub credentials available to the agent if it needs private
#      repos (e.g. via $HF_TOKEN or `hf auth login`).
#
# Usage:
#   ./run.sh --agent <name> [--model <id>] [WORKDIR]
#
# Examples:
#   ./run.sh --agent hermes
#   ./run.sh --agent goose --model gemma-4-26B-A4B-it
#
# ``--agent`` accepts any value listed by ``agentcap run --help``.
# ``--model`` is required for opencode / goose / pi, ignored by hermes.
#
# Env knobs:
#   UPSTREAM        model server URL                http://127.0.0.1:8000
#   LISTEN          proxy bind                      127.0.0.1:8001
#   TURNS           multi-turn count                4
#   FOLLOWUP        continue | templates | synthesized   synthesized
#   SYNTH_UPSTREAM  synth endpoint (bypasses the capture proxy)  $UPSTREAM
#   SYNTH_MODEL     synth model id; auto-detected from
#                   $UPSTREAM/v1/models when unset
#   TIMEOUT         per-turn timeout in seconds     300

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

WORKDIR="${WORKDIR:-$HERE/runs/$AGENT-$(date +%Y-%m-%d-%H%M)}"
UPSTREAM="${UPSTREAM:-http://127.0.0.1:8000}"
LISTEN="${LISTEN:-127.0.0.1:8001}"
TURNS="${TURNS:-4}"
FOLLOWUP="${FOLLOWUP:-synthesized}"
SYNTH_UPSTREAM="${SYNTH_UPSTREAM:-$UPSTREAM}"
SYNTH_MODEL="${SYNTH_MODEL:-$MODEL}"
TIMEOUT="${TIMEOUT:-300}"

if [[ "$FOLLOWUP" == "synthesized" && -z "$SYNTH_MODEL" ]]; then
    SYNTH_MODEL=$(
        curl -sf "$SYNTH_UPSTREAM/v1/models" 2>/dev/null \
            | python3 -c 'import sys,json; d=json.load(sys.stdin); print((d.get("data") or [{}])[0].get("id",""))' \
            || true
    )
    if [[ -z "$SYNTH_MODEL" ]]; then
        echo "ERROR: FOLLOWUP=synthesized requires SYNTH_MODEL; could not auto-detect from $SYNTH_UPSTREAM/v1/models." >&2
        echo "       Set SYNTH_MODEL=<id> or FOLLOWUP=continue." >&2
        exit 2
    fi
    echo "synth model auto-detected: $SYNTH_MODEL" >&2
fi

# Sandbox carries no source code — the corpus is about reaching the
# Hub, not grepping a local checkout. It does carry the AgentSkills
# bundle so cwd-discovery agents (opencode, goose, pi) surface the
# `hf-cli`, `huggingface-datasets`, etc. skills. Hermes uses its own
# ~/.hermes/skills/ and ignores cwd.
mkdir -p "$WORKDIR/sandbox"
SKILLS_SRC="${HF_AGENT_SKILLS_DIR:-$HOME/.agents/skills}"
if [[ -d "$SKILLS_SRC" ]] && [[ ! -e "$WORKDIR/sandbox/skills" ]]; then
    ln -s "$SKILLS_SRC" "$WORKDIR/sandbox/skills"
fi
if [[ ! -f "$WORKDIR/sandbox/AGENTS.md" ]]; then
    curl -sf https://raw.githubusercontent.com/huggingface/skills/main/agents/AGENTS.md \
        > "$WORKDIR/sandbox/AGENTS.md.tmp" \
        && mv "$WORKDIR/sandbox/AGENTS.md.tmp" "$WORKDIR/sandbox/AGENTS.md" \
        || rm -f "$WORKDIR/sandbox/AGENTS.md.tmp"
fi

ARGS=(
    --agent    "$AGENT"
    --upstream "$UPSTREAM"
    --listen   "$LISTEN"
    --tasks    "$HERE/tasks.txt"
    --turns    "$TURNS"
    --followup "$FOLLOWUP"
    --workdir  "$WORKDIR"
    --timeout  "$TIMEOUT"
)
[[ -n "$MODEL" ]] && ARGS+=(--model "$MODEL")
if [[ "$FOLLOWUP" == "synthesized" ]]; then
    ARGS+=(--synth-upstream "$SYNTH_UPSTREAM" --synth-model "$SYNTH_MODEL")
fi

agentcap run "${ARGS[@]}"

echo "done. traces in $WORKDIR/traces, summary in $WORKDIR/run.json"
echo "next: $HERE/export.sh \"$WORKDIR\"   # render + push to the corpus bucket"
