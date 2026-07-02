#!/bin/sh
# Image entrypoint. Substitutes ``AGENTCAP_PROXY_URL`` into the
# baked opencode.json (idempotent) and wires a skills checkout
# into cwd if ``AGENTCAP_SKILLS_DIR`` is set. Uses $HOME (= /root
# in the Linux container via bwrap --setenv; = /home/$user.guest
# under Lima) so the same script works in both backends. Env
# exports duplicate the Containerfile ENV directives so the Lima
# backend (no image-baked ENV) also sees them; redundant under bwrap.
set -e
# Real key when the upstream needs one (HF Router, OpenAI, …); ``dummy``
# is fine for unauthenticated local servers (llama serve, vLLM).
export OPENAI_API_KEY="${AGENTCAP_API_KEY:-dummy}"
export OPENCODE_DISABLE_MODELS_FETCH=1
url="${AGENTCAP_PROXY_URL:-http://127.0.0.1:8001/v1}"
model="${AGENTCAP_MODEL:?AGENTCAP_MODEL is required for opencode}"
sed -i \
    -e "s|@@AGENTCAP_PROXY_URL@@|${url}|g" \
    -e "s|@@AGENTCAP_MODEL@@|${model}|g" \
    "$HOME/.config/opencode/opencode.json"

# Surface opencode's SQLite store on the host. The whole
# ``~/.local/share/opencode/`` dir (opencode.db + WAL/SHM + log/)
# is redirected at AGENTCAP_STATE_DIR/opencode so a crashed
# container leaves a recoverable state. Traces are still
# materialised post-corpus by dump-traces.
if [ -n "${AGENTCAP_STATE_DIR:-}" ] && [ -d "$AGENTCAP_STATE_DIR" ]; then
    mkdir -p "$AGENTCAP_STATE_DIR/opencode"
    mkdir -p "$HOME/.local/share"
    rm -rf "$HOME/.local/share/opencode"
    ln -sfn "$AGENTCAP_STATE_DIR/opencode" "$HOME/.local/share/opencode"
fi

# Skills: AGENTS.md + skills/ symlinked into cwd (where opencode looks).
if [ -n "${AGENTCAP_SKILLS_DIR:-}" ] && [ -d "$AGENTCAP_SKILLS_DIR" ]; then
    [ -f "$AGENTCAP_SKILLS_DIR/agents/AGENTS.md" ] && \
        ln -sfn "$AGENTCAP_SKILLS_DIR/agents/AGENTS.md" "$PWD/AGENTS.md"
    [ -d "$AGENTCAP_SKILLS_DIR/skills" ] && \
        ln -sfn "$AGENTCAP_SKILLS_DIR/skills" "$PWD/skills"
fi

# Toolchain mount (agentcap --tool-dir): put the bundle root and every bin/ it
# ships on PATH, then run its tool_init.sh hook if present (agent-specific setup).
# The dir is bind-mounted (read-only) at its host path.
if [ -n "${AGENTCAP_TOOL_DIR:-}" ] && [ -d "$AGENTCAP_TOOL_DIR" ]; then
    export PATH="$AGENTCAP_TOOL_DIR:$PATH"
    for d in $(find "$AGENTCAP_TOOL_DIR" -maxdepth 2 -type d -name bin 2>/dev/null); do
        export PATH="$d:$PATH"
    done
    [ -f "$AGENTCAP_TOOL_DIR/tool_init.sh" ] && sh "$AGENTCAP_TOOL_DIR/tool_init.sh" || true
fi

# Record this shell's PID so the sandbox can target the about-to-be
# exec'd agent precisely on timeout. ``exec`` keeps $$.
echo $$ > /tmp/agentcap-current.pid
exec "$@"
