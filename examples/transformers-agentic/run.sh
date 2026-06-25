#!/usr/bin/env bash
# Drive the transformers-agentic corpus (the `is-it-agentic-enough` task suite)
# through any registered agent, in one of three assistance tiers. Each task
# names a specific HF model the agent must actually load and run, so the agent
# needs a runnable `transformers` — provided by a self-contained, relocatable
# bundle mounted via `agentcap run --tool-dir` (build it once with
# ./build-toolenv.sh). The agent's own model is served on $UPSTREAM as usual.
#
# Tiers (the article's bare/clone/skill discovery conditions):
#   bare   empty cwd; only the mounted transformers bundle is available.
#   clone  cwd is a detached git worktree of ./transformers @ the bundle's
#          commit, so AGENTS.md / cli/agentic/*.py auto-discover from cwd.
#   skill  empty cwd + the packaged transformers Skill (./skill) in context.
#
# Usage:
#   ./run.sh --agent <name> --model <id> [--tier bare|clone|skill] [--tasks <file>]
#
# Examples:
#   ./run.sh --agent pi      --model unsloth/GLM-4.5-Air-GGUF --tier skill
#   ./run.sh --agent hermes  --model unsloth/GLM-4.5-Air-GGUF --tier bare
#
# Captures land under $HERE/.agentcap/<run-id>/; publish with `agentcap export`.
#
# Env knobs:
#   UPSTREAM   model server URL                http://127.0.0.1:8001
#   TURNS      turns per task                  1
#   FOLLOWUP   continue | templates | synthesized   continue
#   TIMEOUT    per-turn timeout (seconds)      900

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
export AGENTCAP_WORKSPACE="$HERE"

AGENT="" MODEL="" TIER="bare" TASKS="$HERE/tasks.txt"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --agent) AGENT="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --tier)  TIER="$2";  shift 2 ;;
        --tasks) TASKS="$2"; shift 2 ;;
        -h|--help) sed -n '/^# Usage:/,/^set -euo/p' "$0" | sed 's/^# \?//; /^set -euo/d'; exit 0 ;;
        *) echo "ERROR: unexpected arg: $1" >&2; exit 2 ;;
    esac
done
[[ -n "$AGENT" && -n "$MODEL" ]] || { echo "ERROR: --agent and --model are required. See: $0 --help" >&2; exit 2; }
[[ "$TIER" =~ ^(bare|clone|skill)$ ]] || { echo "ERROR: --tier must be bare|clone|skill" >&2; exit 2; }
[[ -f "$TASKS" ]] || { echo "ERROR: tasks file not found: $TASKS" >&2; exit 2; }

UPSTREAM="${UPSTREAM:-http://127.0.0.1:8001}"
TURNS="${TURNS:-1}"
FOLLOWUP="${FOLLOWUP:-continue}"
TIMEOUT="${TIMEOUT:-900}"

TOOLENV="$HERE/toolenv"
[[ -x "$TOOLENV/bin/transformers" ]] || {
    echo "ERROR: transformers bundle missing at $TOOLENV. Build it first:" >&2
    echo "         ./build-toolenv.sh" >&2
    exit 2
}

# Per-tier sandbox cwd, rebuilt fresh each invocation, with inputs/ seeded.
SANDBOX="$HERE/sandbox-$TIER"
if [[ -e "$SANDBOX/.git" ]]; then
    git -C "$HERE/transformers" worktree remove --force "$SANDBOX" 2>/dev/null || true
fi
rm -rf "$SANDBOX"
if [[ "$TIER" == "clone" ]]; then
    [[ -d "$HERE/transformers/.git" ]] || { echo "ERROR: clone tier needs $HERE/transformers (built by ./build-toolenv.sh)" >&2; exit 2; }
    SHA="$(git -C "$HERE/transformers" rev-parse HEAD)"
    git -C "$HERE/transformers" worktree add --detach "$SANDBOX" "$SHA" >/dev/null
else
    mkdir -p "$SANDBOX"
fi
cp -r "$HERE/inputs" "$SANDBOX/inputs"

# Only the skill tier passes --skills; empty otherwise. Expanded set-u-safe below
# (bash 3.2 treats "${arr[@]}" of an empty array as an unbound-variable error).
skill_args=()
[[ "$TIER" == "skill" ]] && skill_args=(--skills "$HERE/skill")

echo ">>> agent=$AGENT model=$MODEL tier=$TIER tasks=$(basename "$TASKS") upstream=$UPSTREAM" >&2
agentcap run \
    --agent     "$AGENT" \
    --model     "$MODEL" \
    --upstream  "$UPSTREAM" \
    --sandbox   "$SANDBOX" \
    --tool-dir  "$TOOLENV" \
    --label     "$TIER" \
    "${skill_args[@]+"${skill_args[@]}"}" \
    --tasks     "$TASKS" \
    --turns     "$TURNS" \
    --followup  "$FOLLOWUP" \
    --timeout   "$TIMEOUT"

echo "done. captures under $HERE/.agentcap/ (agentcap ls). publish: agentcap export" >&2
