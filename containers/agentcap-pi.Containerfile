# Per-agent sandbox image for the pi-coding-agent CLI
# (``@mariozechner/pi-coding-agent`` on npm).
#
# Pinning ``PI_VERSION`` is the reproducibility knob — bump it
# deliberately, never let it drift to ``latest``.

FROM ubuntu:24.04

# pi-tui's regex uses the ECMAScript 2024 `v` flag, which Ubuntu's
# default Node 18 doesn't grok ("SyntaxError: Invalid regular
# expression flags"). Pin to NodeSource's Node 20 LTS line.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        ca-certificates curl python3 python3-pip gnupg \
 && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
 && apt-get install -y --no-install-recommends nodejs \
 && rm -rf /var/lib/apt/lists/*

ARG PI_VERSION=0.73.0
RUN npm install -g --no-fund --no-audit \
        "@mariozechner/pi-coding-agent@${PI_VERSION}" \
 && command -v pi

# Canonical Hub interaction tool — see agentcap-goose.Containerfile
# for rationale. Same install across all four images.
RUN pip3 install --break-system-packages --no-cache-dir 'huggingface_hub[cli]' \
 && command -v hf

# Bake the per-run config the driver previously wrote. The proxy
# URL is templated as ``@@AGENTCAP_PROXY_URL@@`` and rendered by the
# entrypoint script using ``AGENTCAP_PROXY_URL`` at run time. pi
# validates ``--model`` against the ``models`` array in models.json,
# so we bake at least gemma-4-E4B-it (the default test target). Add
# more models + rebuild to test others with pi.
ENV PI_CODING_AGENT_DIR=/opt/pi-config \
    PI_CODING_AGENT_SESSION_DIR=/opt/pi-config/sessions \
    PI_OFFLINE=1 \
    PI_SKIP_VERSION_CHECK=1 \
    PI_LOCAL_API_KEY=dummy
RUN mkdir -p /opt/pi-config/sessions
COPY agentcap-pi/models.json /opt/pi-config/models.json
COPY agentcap-pi/agentcap-init.sh /usr/local/bin/agentcap-init
RUN chmod 0755 /usr/local/bin/agentcap-init
ENTRYPOINT ["/usr/local/bin/agentcap-init"]
