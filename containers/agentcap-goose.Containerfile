# Per-agent sandbox image for the goose CLI.
#
# Built once by `agentcap run --agent goose` (or eagerly via
# `agentcap.sandbox.image_provisioning.ensure_image("goose")`). The
# resulting image is mounted as the bwrap rootfs at run time — the
# agent binary lives in the image, not on the host.
#
# Mirror of scripts/lima/agentcap-goose.yaml (same install command,
# different runtime).

FROM ubuntu:24.04

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        ca-certificates curl python3 python3-pip bzip2 xz-utils libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Goose's install script lands the binary in ``$HOME/.local/bin/goose``.
# Run as root, then copy into /usr/local/bin so the agent is on PATH
# regardless of which $HOME the runtime mounts.
RUN curl -fsSL https://github.com/block/goose/releases/download/stable/download_cli.sh \
        | CONFIGURE=false bash \
 && install -m 0755 /root/.local/bin/goose /usr/local/bin/goose \
 && command -v goose

# Canonical Hub interaction tool. ``huggingface_hub[cli]`` ships the
# ``hf`` binary used by skills, the orchestrator, and ad-hoc agent
# commands (auth login, upload, model queries). Baked in every image
# regardless of corpus — it's a small Python install (~50 MB) and
# part of the standard agent ecosystem.
RUN pip3 install --break-system-packages --no-cache-dir 'huggingface_hub[cli]' \
 && command -v hf

# Goose talks to an OpenAI-compat backend via env vars. The
# in-process proxy URL comes from ``AGENTCAP_PROXY_URL`` at run
# time (default = the in-process proxy on :8001); the entrypoint
# script translates it into ``OPENAI_HOST``.
ENV OPENAI_API_KEY=dummy \
    GOOSE_PROVIDER=openai

COPY agentcap-goose/agentcap-init.sh /usr/local/bin/agentcap-init
RUN chmod 0755 /usr/local/bin/agentcap-init
ENTRYPOINT ["/usr/local/bin/agentcap-init"]
