#!/bin/sh
# Image entrypoint. Translates ``AGENTCAP_PROXY_URL`` (full /v1 URL)
# into goose's expected ``OPENAI_HOST`` (host root, no /v1) and
# wires a skills checkout into cwd if ``AGENTCAP_SKILLS_DIR`` is set.
# Default proxy URL points at the agentcap in-process proxy.
set -e
url="${AGENTCAP_PROXY_URL:-http://127.0.0.1:8001/v1}"
export OPENAI_HOST="${url%/v1}"

# Skills: AGENTS.md + skills/ symlinked into cwd (where goose looks).
if [ -n "${AGENTCAP_SKILLS_DIR:-}" ] && [ -d "$AGENTCAP_SKILLS_DIR" ]; then
    [ -f "$AGENTCAP_SKILLS_DIR/agents/AGENTS.md" ] && \
        ln -sfn "$AGENTCAP_SKILLS_DIR/agents/AGENTS.md" "$PWD/AGENTS.md"
    [ -d "$AGENTCAP_SKILLS_DIR/skills" ] && \
        ln -sfn "$AGENTCAP_SKILLS_DIR/skills" "$PWD/skills"
fi

exec "$@"
