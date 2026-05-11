#!/bin/sh
# Image entrypoint. Substitutes ``AGENTCAP_PROXY_URL`` into the
# baked models.json (idempotent — re-running on a rendered file is
# a no-op) and wires a skills checkout into cwd if
# ``AGENTCAP_SKILLS_DIR`` is set.
set -e
url="${AGENTCAP_PROXY_URL:-http://127.0.0.1:8001/v1}"
sed -i "s|@@AGENTCAP_PROXY_URL@@|${url}|g" /opt/pi-config/models.json

# Skills: AGENTS.md + skills/ symlinked into cwd (where pi looks).
if [ -n "${AGENTCAP_SKILLS_DIR:-}" ] && [ -d "$AGENTCAP_SKILLS_DIR" ]; then
    [ -f "$AGENTCAP_SKILLS_DIR/agents/AGENTS.md" ] && \
        ln -sfn "$AGENTCAP_SKILLS_DIR/agents/AGENTS.md" "$PWD/AGENTS.md"
    [ -d "$AGENTCAP_SKILLS_DIR/skills" ] && \
        ln -sfn "$AGENTCAP_SKILLS_DIR/skills" "$PWD/skills"
fi

exec "$@"
