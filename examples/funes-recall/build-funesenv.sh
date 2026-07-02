#!/usr/bin/env bash
# One-time builder for the self-contained `funes` bundle mounted via
# `agentcap run --tool-dir`. Contents:
#
#   bin/funes        wrapper: sets FASTEMBED_CACHE_DIR + FUNES_HOME from its own dir,
#                    then exec's funes.real (self-configuring wherever it's mounted)
#   bin/funes.real   the funes-<arch>-linux release binary
#   cache/           prewarmed fastembed cache (bge-small + bge-reranker-base)
#   funes-home/      FUNES_HOME: funes.json → the shared hf:// index (read live)
#   tool_init.sh     wiring agentcap runs in the sandbox: `funes install <agent>`,
#                    which installs funes into whichever agent is running
#
# Re-run to refresh; the binary fetch and prewarm are skipped if already present.
# Requires `curl`. No HF token needed for the default (public) index; a private
# FUNES_REMOTE needs one (`HF_TOKEN`, or ~/.cache/huggingface/token).

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ENV_DIR="$HERE/funesenv"

# Public HF bucket serving the latest prebuilt funes binaries.
FUNES_BUCKET="${FUNES_BUCKET:-https://huggingface.co/buckets/huggingface/funes}"
# The bundle runs inside the agent sandbox, whose arch matches this host's. Pick the
# matching binary so it's native (podman does no cross-arch emulation).
case "$(uname -m)" in
    x86_64|amd64)   ASSET="funes-x86_64-linux" ;;
    aarch64|arm64)  ASSET="funes-aarch64-linux" ;;
    *) echo "ERROR: no funes binary for arch $(uname -m)" >&2; exit 2 ;;
esac
# The shared funes index this bundle recalls from, as an `<org>/<repo>` HF dataset.
FUNES_REMOTE="${FUNES_REMOTE:-dacorvo/funes-bench}"

mkdir -p "$ENV_DIR/bin" "$ENV_DIR/cache" "$ENV_DIR/funes-home"

# 1. The funes binary — the latest, pulled straight from the public HF bucket.
if [ ! -x "$ENV_DIR/bin/funes.real" ]; then
    echo ">>> fetching $ASSET from $FUNES_BUCKET"
    curl -fsSL "$FUNES_BUCKET/resolve/$ASSET" -o "$ENV_DIR/bin/funes.real"
    chmod +x "$ENV_DIR/bin/funes.real"
fi

# 2. Self-configuring wrapper: points FASTEMBED_CACHE_DIR + FUNES_HOME at the bundle
#    relative to its own dir (no per-sandbox env needed), and reads an HF token dropped
#    into the bundle if present (only a private FUNES_REMOTE needs one). HOME is left for
#    lance's download cache.
cat > "$ENV_DIR/bin/funes" <<'WRAP'
#!/bin/sh
BIN="$(cd "$(dirname "$0")" && pwd)"
export FASTEMBED_CACHE_DIR="$BIN/../cache"
export FUNES_HOME="$BIN/../funes-home"
TOK="$BIN/../funes-home/.cache/huggingface/token"
[ -z "${HF_TOKEN:-}" ] && [ -f "$TOK" ] && HF_TOKEN="$(cat "$TOK")" && export HF_TOKEN
exec "$BIN/funes.real" "$@"
WRAP
chmod +x "$ENV_DIR/bin/funes"

# 2b. Wiring agentcap runs in the sandbox (it sets AGENTCAP_AGENT): funes installs itself
#     into whichever agent is running — a bridge extension for pi, an MCP server for
#     hermes/opencode.
cat > "$ENV_DIR/tool_init.sh" <<'HOOK'
#!/bin/sh
# FUNES_BIN is the bundle's self-configuring wrapper (sets FUNES_HOME + the prewarmed
# cache). hermes/opencode bake it into their MCP config; pi's extension finds the same
# wrapper on PATH.
export FUNES_BIN="$AGENTCAP_TOOL_DIR/bin/funes"
"$FUNES_BIN" install "$AGENTCAP_AGENT" >/dev/null 2>&1 || true
HOOK
chmod +x "$ENV_DIR/tool_init.sh"

# 3. FUNES_HOME: recall reads the shared index live over hf:// (no local store).
cat > "$ENV_DIR/funes-home/funes.json" <<JSON
{
  "remote": "hf://datasets/$FUNES_REMOTE"
}
JSON

# 4. Prewarm the embedder + reranker, inside ubuntu:24.04 (the agent-image base) so the
#    native binary writes a cache it can read back from the read-only mount. Any recall
#    loads both models, so a throwaway query warms the cache even if it returns nothing.
if [ -z "$(ls -A "$ENV_DIR/cache" 2>/dev/null)" ]; then
    echo ">>> prewarming embedder + reranker into the bundle cache"
    podman run --rm -i \
        -e HF_TOKEN="${HF_TOKEN:-}" \
        -v "$ENV_DIR:$ENV_DIR" -w "$ENV_DIR" \
        ubuntu:24.04 sh -s <<IN_CONTAINER
set -e
apt-get update -q >/dev/null
apt-get install -y -q --no-install-recommends ca-certificates >/dev/null
export FASTEMBED_CACHE_DIR="$ENV_DIR/cache" FUNES_HOME="$ENV_DIR/funes-home"
"$ENV_DIR/bin/funes.real" recall warmup --k 1 >/dev/null 2>&1 || true
[ -n "\$(ls -A "$ENV_DIR/cache" 2>/dev/null)" ] || { echo "    prewarm produced no cache — check network" >&2; exit 1; }
IN_CONTAINER
fi

echo ">>> DONE. bundle at $ENV_DIR ($(du -sh "$ENV_DIR" | cut -f1))"
echo "    mount it with: agentcap run --agent pi --tool-dir $ENV_DIR ..."
