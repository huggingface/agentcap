# Per-agent sandbox image for the hermes-agent CLI (Nous Research).
#
# Capture flow only — agentcap exercises ``hermes chat -q "<prompt>"``,
# which is the Python entrypoint. The upstream Dockerfile also bundles
# a browser UI (Playwright) and a TUI (npm); both are optional for the
# capture path and intentionally skipped here. If you need them,
# extend this file or run agents from the upstream image directly.

FROM ubuntu:24.04

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        ca-certificates curl git \
        python3 python3-venv python3-pip \
 && rm -rf /var/lib/apt/lists/*

# ``HERMES_REF`` accepts a branch, tag, or commit SHA.
ARG HERMES_REF=main
RUN git clone https://github.com/NousResearch/hermes-agent.git /opt/hermes \
 && git -C /opt/hermes checkout "${HERMES_REF}" \
 && python3 -m venv /opt/hermes/venv \
 && /opt/hermes/venv/bin/pip install --no-cache-dir -e "/opt/hermes[all]" \
 && ln -s /opt/hermes/venv/bin/hermes /usr/local/bin/hermes \
 && command -v hermes

# Canonical Hub interaction tool — see agentcap-goose.Containerfile
# for rationale. Installs system-wide (separate from hermes's venv).
RUN pip3 install --break-system-packages --no-cache-dir 'huggingface_hub[cli]' \
 && command -v hf

# Bootstrap ``~/.hermes/`` with the production config baked in.
# ``hermes --version`` populates the identity files (SOUL.md,
# memories/, sessions/, logs/, …); the ``config set`` calls
# materialise ``config.yaml`` pointing at the fixed in-process
# proxy URL. The driver no longer rewrites this at runtime.
# ``auxiliary.compression.context_length`` is set too — hermes
# refuses startup if either guard is below 64K.
RUN hermes --version >/dev/null \
 && hermes config set model.provider custom \
 && hermes config set model.base_url http://placeholder/v1 \
 && hermes config set model.context_length 65536 \
 && hermes config set auxiliary.compression.context_length 65536 \
 && test -f /root/.hermes/config.yaml

# Wire hermes's bundled skill catalog (~25 entries under
# /opt/hermes/skills/) into ~/.hermes/skills/. We need a REAL
# directory here, not a symlink to /opt/hermes/skills, so the
# entrypoint can add user-provided skills (via --skills) as
# additional symlinks alongside the bundled ones.
RUN mkdir -p /root/.hermes/skills \
 && for d in /opt/hermes/skills/*/; do \
        ln -sfn "$d" "/root/.hermes/skills/$(basename "$d")"; \
    done \
 && hermes skills list 2>&1 | grep -E "^[0-9]+ hub" >/dev/null

# Entrypoint: the script overrides model.base_url from
# ``AGENTCAP_PROXY_URL`` at run time (default = the in-process
# proxy on :8001). ``hermes config set`` round-trips the YAML, so
# we don't sed.
COPY agentcap-hermes/agentcap-init.sh /usr/local/bin/agentcap-init
RUN chmod 0755 /usr/local/bin/agentcap-init
ENTRYPOINT ["/usr/local/bin/agentcap-init"]
