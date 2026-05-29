#!/usr/bin/env bash
# Drive the hf-hub-session corpus through any registered agent. The
# corpus is 30 open-ended exploration prompts over the Hugging Face
# Hub (models / datasets / spaces / papers) — the agent is expected
# to reach for the `hf` CLI, the `huggingface_hub` SDK, or direct
# HTTPS calls to https://huggingface.co/api/... as its primary tool
# surface. The sandbox carries the huggingface/skills bundle so the
# agent can `skill_view(hf-cli)` etc.; without it the loop ends in
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
#      on first run. Override with SKILLS_CHECKOUT=<path>. Expected
#      layout: <checkout>/agents/AGENTS.md (catalog) + <checkout>/skills/
#      (bundle dirs).
#
# Per-agent sandbox setup:
#   * opencode / goose / pi (cwd-discovery): the script seeds the
#     per-run sandbox with symlinks into the SKILLS_CHECKOUT:
#       <sandbox>/AGENTS.md  -> <checkout>/agents/AGENTS.md
#       <sandbox>/skills/    -> <checkout>/skills/
#   * hermes (ignores cwd; reads HERMES_HOME/skills/): no injection
#     from this script — install HF skills into the agentcap-hermes
#     VM as part of provisioning (or post-hoc via
#     `hermes skills install`). The driver snapshots whatever the
#     sandbox's ~/.hermes/skills/ contains into the per-run
#     HERMES_HOME overlay; it does not accept a host-side extra-
#     skills path anymore.
#
# Usage:
#   ./run.sh --agent <name> --model <id> [WORKDIR]
#
# Examples:
#   ./run.sh --agent hermes --model Qwen/Qwen3-8B
#   ./run.sh --agent goose --model gemma-4-26B-A4B-it
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
#   SKILLS_CHECKOUT path to a huggingface/skills clone (see Prereqs.4)

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

# Resolve a huggingface/skills checkout. The corpus is about reaching
# the Hub via documented skills (hf-cli, huggingface-datasets, etc.),
# so a missing skills bundle is fatal: agents will spin in
# "skill not found" with no fallback. Mirrors the
# transformers-coding-session pattern of an explicit local checkout.
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
# Repo layout: agents/AGENTS.md is the catalog; the skill bundles live
# under skills/<name>/SKILL.md at the repo root (the catalog's paths
# are relative to wherever AGENTS.md is loaded, so AGENTS.md and
# skills/ must sit side-by-side in the sandbox).
SKILLS_DIR="$SKILLS_CHECKOUT/skills"
AGENTS_MD="$SKILLS_CHECKOUT/agents/AGENTS.md"
if [[ ! -d "$SKILLS_DIR" || ! -f "$AGENTS_MD" ]]; then
    echo "ERROR: SKILLS_CHECKOUT=$SKILLS_CHECKOUT does not look like a huggingface/skills clone." >&2
    echo "       Expected skills/ and agents/AGENTS.md under it." >&2
    echo "       Either point SKILLS_CHECKOUT at a valid clone, or run:" >&2
    echo "         git clone https://github.com/huggingface/skills \$HOME/skills" >&2
    exit 2
fi
echo "[skills] using checkout: $SKILLS_CHECKOUT" >&2

# Pass the skills checkout to `agentcap run --skills`. The per-agent
# image entrypoint bind-mounts it read-only and wires it into the
# agent's expected discovery location:
#   * hermes: ~/.hermes/skills (skill_view tool) + cwd/AGENTS.md
#   * opencode/goose/pi: cwd/AGENTS.md + cwd/skills
SANDBOX="$WORKDIR/sandbox"
mkdir -p "$SANDBOX"

ARGS=(
    --agent     "$AGENT"
    --upstream  "$UPSTREAM"
    --workspace "$SANDBOX"
    --skills    "$SKILLS_CHECKOUT"
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
echo "next: $HERE/export.sh \"$WORKDIR\"   # render + push to the corpus dataset"
