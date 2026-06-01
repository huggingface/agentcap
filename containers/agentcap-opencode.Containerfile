# Per-agent sandbox image for the opencode CLI.
#
# Built once by `agentcap run --agent opencode`. Mirror of
# scripts/lima/agentcap-opencode.yaml.

FROM ubuntu:24.04

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        ca-certificates curl python3 python3-pip unzip \
 && rm -rf /var/lib/apt/lists/*

# opencode's installer drops its tree under ``$HOME/.opencode/``. Move
# the whole tree under /opt/ (out of the per-run $HOME tmpfs) and
# symlink the launcher into /usr/local/bin so it's on PATH. Pin the
# release explicitly (the installer otherwise grabs latest at build
# time, making the image hash non-deterministic).
ARG OPENCODE_VERSION=1.15.13
RUN curl -fsSL https://opencode.ai/install \
        | bash -s -- --version "${OPENCODE_VERSION}" \
 && mv /root/.opencode /opt/opencode \
 && ln -s /opt/opencode/bin/opencode /usr/local/bin/opencode \
 && command -v opencode

# Canonical Hub interaction tool — see agentcap-goose.Containerfile
# for rationale. Same install across all four images.
RUN pip3 install --break-system-packages --no-cache-dir 'huggingface_hub[cli]' \
 && command -v hf

# Bake the opencode.json the driver previously wrote per run. The
# provider URL is templated as ``@@AGENTCAP_PROXY_URL@@`` and rendered
# by the entrypoint script using ``AGENTCAP_PROXY_URL`` at run time.
# opencode validates ``--model local/<id>`` against the ``models``
# block, so we have to bake at least one model id;
# ``gemma-4-E4B-it`` matches the default test target. Add more models
# to opencode.json and rebuild to test other models. The ``minimal``
# agent (stripped prompt + read/edit only) is pre-defined for
# CPU + small-model runs.
ENV OPENAI_API_KEY=dummy \
    OPENCODE_DISABLE_MODELS_FETCH=1
RUN mkdir -p /root/.config/opencode
COPY agentcap-opencode/opencode.json /root/.config/opencode/opencode.json
COPY agentcap-opencode/agentcap-init.sh /usr/local/bin/agentcap-init
COPY agentcap-opencode/dump-traces.sh /usr/local/bin/dump-traces
RUN chmod 0755 /usr/local/bin/agentcap-init /usr/local/bin/dump-traces
ENTRYPOINT ["/usr/local/bin/agentcap-init"]
