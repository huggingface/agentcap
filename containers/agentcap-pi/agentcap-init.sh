#!/bin/sh
# Image entrypoint. Substitutes ``AGENTCAP_PROXY_URL`` into the
# baked models.json (idempotent — re-running on a rendered file is
# a no-op) and wires a skills checkout into cwd if
# ``AGENTCAP_SKILLS_DIR`` is set. Env exports duplicate the
# Containerfile ENV directives so the Lima backend (no image-baked
# ENV) also sees them; redundant under bwrap.
set -e
export PI_CODING_AGENT_DIR=/opt/pi-config
export PI_CODING_AGENT_SESSION_DIR=/opt/pi-config/sessions
export PI_OFFLINE=1
export PI_SKIP_VERSION_CHECK=1
export PI_LOCAL_API_KEY=dummy
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
