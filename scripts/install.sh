#!/bin/sh
# Install the agentcap binary from GitHub Releases.
#
#   curl -fsSL https://raw.githubusercontent.com/huggingface/agentcap/main/scripts/install.sh | sh
#
# Detects the platform, downloads the matching prebuilt binary, and installs it
# onto your PATH. Flags (pass after `sh -s --` when piping):
#   -b <dir>   install dir          (default: $HOME/.local/bin; env: AGENTCAP_INSTALL_DIR)
#   -v <tag>   release tag to fetch (default: latest;           env: AGENTCAP_VERSION)
set -eu

REPO="huggingface/agentcap"
BINDIR="${AGENTCAP_INSTALL_DIR:-$HOME/.local/bin}"
VERSION="${AGENTCAP_VERSION:-latest}"

usage() {
    echo "usage: install.sh [-b install-dir] [-v release-tag]" >&2
    exit "${1:-2}"
}

while getopts "b:v:h" opt; do
    case "$opt" in
        b) BINDIR="$OPTARG" ;;
        v) VERSION="$OPTARG" ;;
        h) usage 0 ;;
        *) usage 2 ;;
    esac
done

# (OS, arch) -> the asset name the release workflow publishes. Only these two
# targets are built; everything else falls through to build-from-source.
case "$(uname -s)-$(uname -m)" in
    Linux-x86_64)                  asset="agentcap-x86_64-linux" ;;
    Darwin-arm64 | Darwin-aarch64) asset="agentcap-arm64-apple-darwin" ;;
    *)
        echo "agentcap: no prebuilt binary for $(uname -s)/$(uname -m)." >&2
        echo "Build from source: https://github.com/$REPO#building-from-source" >&2
        exit 1
        ;;
esac

if [ "$VERSION" = latest ]; then
    url="https://github.com/$REPO/releases/latest/download/$asset"
else
    url="https://github.com/$REPO/releases/download/$VERSION/$asset"
fi

# Download to a temp file so a failed/partial fetch never lands on PATH.
tmp="$(mktemp 2>/dev/null || mktemp -t agentcap)"
trap 'rm -f "$tmp"' EXIT INT TERM

if command -v curl >/dev/null 2>&1; then
    fetch() { curl -fsSL "$1" -o "$2"; }
elif command -v wget >/dev/null 2>&1; then
    fetch() { wget -qO "$2" "$1"; }
else
    echo "agentcap: need curl or wget on PATH to download." >&2
    exit 1
fi

echo "Downloading $asset ($VERSION)…"
if ! fetch "$url" "$tmp"; then
    echo "agentcap: download failed: $url" >&2
    echo "  (no matching release published yet? see https://github.com/$REPO/releases)" >&2
    exit 1
fi

mkdir -p "$BINDIR"
chmod +x "$tmp"
mv "$tmp" "$BINDIR/agentcap"
trap - EXIT INT TERM

echo "Installed agentcap -> $BINDIR/agentcap"
"$BINDIR/agentcap" --version 2>/dev/null || true

case ":$PATH:" in
    *":$BINDIR:"*) ;;
    *)
        echo
        echo "$BINDIR is not on your PATH. Add it:"
        echo "  export PATH=\"$BINDIR:\$PATH\""
        ;;
esac
