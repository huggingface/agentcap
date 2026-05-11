#!/bin/sh
# Image entrypoint. Substitutes ``AGENTCAP_PROXY_URL`` into the
# baked opencode.json (idempotent) and wires a skills checkout
# into cwd if ``AGENTCAP_SKILLS_DIR`` is set. Uses $HOME (= /root
# in the Linux container via bwrap --setenv; = /home/$user.guest
# under Lima) so the same script works in both backends. Env
# exports duplicate the Containerfile ENV directives so the Lima
# backend (no image-baked ENV) also sees them; redundant under bwrap.
set -e
export OPENAI_API_KEY=dummy
export OPENCODE_DISABLE_MODELS_FETCH=1
url="${AGENTCAP_PROXY_URL:-http://127.0.0.1:8001/v1}"
sed -i "s|@@AGENTCAP_PROXY_URL@@|${url}|g" \
    "$HOME/.config/opencode/opencode.json"

# Skills: AGENTS.md + skills/ symlinked into cwd (where opencode looks).
if [ -n "${AGENTCAP_SKILLS_DIR:-}" ] && [ -d "$AGENTCAP_SKILLS_DIR" ]; then
    [ -f "$AGENTCAP_SKILLS_DIR/agents/AGENTS.md" ] && \
        ln -sfn "$AGENTCAP_SKILLS_DIR/agents/AGENTS.md" "$PWD/AGENTS.md"
    [ -d "$AGENTCAP_SKILLS_DIR/skills" ] && \
        ln -sfn "$AGENTCAP_SKILLS_DIR/skills" "$PWD/skills"
fi

exec "$@"
