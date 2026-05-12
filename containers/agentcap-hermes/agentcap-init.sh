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

# Provider switch driven by the orchestrator's upstream probe
# (AGENTCAP_PROVIDER). For hosted providers we use the built-in
# profile that reads its API key from os.environ (env_vars=...)
# — credentials flow through process env only, never get
# persisted to ~/.hermes/config.yaml. For local servers
# (llama.cpp / vLLM / unknown) ``custom`` is the right profile —
# no auth needed, the proxy URL is the only knob.
case "${AGENTCAP_PROVIDER:-custom}" in
    hf-router|hf-router/*)
        hermes config set model.provider huggingface >/dev/null
        [ -n "${AGENTCAP_API_KEY:-}" ] && export HF_TOKEN="$AGENTCAP_API_KEY"
        ;;
    openai|openai/*)
        hermes config set model.provider openai >/dev/null
        [ -n "${AGENTCAP_API_KEY:-}" ] && export OPENAI_API_KEY="$AGENTCAP_API_KEY"
        ;;
    *)
        hermes config set model.provider custom >/dev/null
        ;;
esac

# Model id flows via the hermes CLI ``-m`` flag (the driver
# appends it) — config-side model.name is left alone.

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
