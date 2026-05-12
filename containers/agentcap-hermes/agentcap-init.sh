#!/bin/sh
# Image entrypoint. Sets hermes's ``model.base_url`` to
# ``AGENTCAP_PROXY_URL`` (uses ``hermes config set`` to round-trip
# the YAML) and adds any user-provided skills to the bundled catalog.
#
# Skills wiring:
#  * The image already baked ~/.hermes/skills/ as a real dir with
#    symlinks to every bundled entry under /opt/hermes/skills/.
#  * When --skills is set on the agentcap CLI, AGENTCAP_SKILLS_DIR
#    points at a host dir whose ``skills/`` subdir holds extra
#    skill bundles. We add a symlink per bundle into
#    ~/.hermes/skills/ — ADDITIVE, not a replacement. Name
#    collisions with bundled skills overwrite (user wins).
#  * AGENTS.md from the skill checkout is exposed in cwd because
#    hermes auto-injects it into the system prompt.
set -e
url="${AGENTCAP_PROXY_URL:-http://127.0.0.1:8001/v1}"
hermes config set model.base_url "$url" >/dev/null

# Without an explicit model.name, hermes falls back to its built-in
# default (currently ``google/gemma-4-E4B-it``), which is what gets
# sent as the ``model`` field on every outbound request and recorded
# by the capture proxy. Tying it to AGENTCAP_MODEL keeps captured
# traces honest about which model the agent was actually run against.
if [ -n "${AGENTCAP_MODEL:-}" ]; then
    hermes config set model.name "$AGENTCAP_MODEL" >/dev/null
fi

if [ -n "${AGENTCAP_SKILLS_DIR:-}" ] && [ -d "$AGENTCAP_SKILLS_DIR" ]; then
    if [ -d "$AGENTCAP_SKILLS_DIR/skills" ]; then
        mkdir -p "$HOME/.hermes/skills"
        for d in "$AGENTCAP_SKILLS_DIR/skills/"*/; do
            [ -d "$d" ] && \
                ln -sfn "$d" "$HOME/.hermes/skills/$(basename "$d")"
        done
    fi
    [ -f "$AGENTCAP_SKILLS_DIR/agents/AGENTS.md" ] && \
        ln -sf "$AGENTCAP_SKILLS_DIR/agents/AGENTS.md" "$PWD/AGENTS.md"
fi

exec "$@"
