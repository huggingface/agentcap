#!/usr/bin/env bash
# One-time builder for the self-contained `transformers` bundle the corpus
# mounts via `agentcap run --tool-dir`. Everything (interpreter, torch, the
# agentic-CLI transformers, and a prewarmed model cache) lives under ./toolenv,
# built INSIDE ubuntu:24.04 — the exact base of every agentcap agent image — so
# the venv's /usr/bin/python3.12 base and torch .so's are ABI-identical when the
# bundle is mounted (read-only) into any agent sandbox.
#
#   ./toolenv/                relocatable venv (bin/transformers, bin/python, lib/)
#   ./toolenv/hf-cache/       prewarmed HF cache; the venv points HF_HOME here
#   ./transformers/           transformers checkout @ PINNED_SHA (clone-tier source)
#   ./inputs/                 corpus inputs (cat.jpg, sample.wav), fetched from the blog repo
#
# Re-run to prewarm any missing models; the heavy build is skipped if ./toolenv
# already has a working `transformers`. Pass HF_TOKEN for faster, rate-limit-free
# downloads. The agentic CLI is unreleased, so we pin the exact commit.

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

# transformers @ the "agent-first CLI" effort (is-it-agentic-enough's
# `4d15b215f3` / "w/ CLI + Skill"). Not on main, not in any release.
PINNED_SHA="4d15b215f37bcb25a2d6472b2147e34b3d465186"

# Every model named in the corpus (tasks.txt). Prewarmed so runs read offline.
MODELS="
distilbert/distilbert-base-uncased-finetuned-sst-2-english
dslim/bert-base-NER
openai/whisper-tiny
llava-hf/llava-interleave-qwen-0.5b-hf
HuggingFaceTB/SmolLM2-360M-Instruct
facebook/bart-large-cnn
distilbert/distilbert-base-cased-distilled-squad
distilbert/distilbert-base-uncased
facebook/bart-large-mnli
google/vit-base-patch16-224
facebook/detr-resnet-50
laion/clap-htsat-unfused
Helsinki-NLP/opus-mt-en-fr
"

# Corpus inputs (the cat image + audio clip the tasks reference) live in the
# is-it-agentic-enough repo; fetch them at a pinned commit instead of vendoring
# binaries here. Idempotent — skipped if already present.
INPUTS_SHA="1655d61abf056c58ee2bc8682cb2f0d336ce31ae"
INPUTS_URL="https://raw.githubusercontent.com/huggingface/is-it-agentic-enough/${INPUTS_SHA}/src/ae/data/inputs"
mkdir -p "$HERE/inputs"
for f in cat.jpg sample.wav; do
    [ -f "$HERE/inputs/$f" ] || { echo ">>> fetching input $f"; curl -fsSL "$INPUTS_URL/$f" -o "$HERE/inputs/$f"; }
done

podman run --rm -i \
    -e PINNED_SHA="$PINNED_SHA" -e MODELS="$MODELS" \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -e HF_HUB_DISABLE_XET=1 \
    -v "$HERE:$HERE" -w "$HERE" \
    ubuntu:24.04 bash -s <<'IN_CONTAINER'
set -e
export DEBIAN_FRONTEND=noninteractive
apt-get update -q >/dev/null
apt-get install -y -q --no-install-recommends python3 python3-venv python3-pip git ca-certificates >/dev/null
HERE="$(pwd)"; TE="$HERE/toolenv"; TFSRC="$HERE/transformers"
# Prewarm into the bundle's cache, explicitly online (the .pth written below
# makes the venv default to offline at run time; setdefault leaves these be).
export HF_HOME="$TE/hf-cache" HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0

if [ ! -x "$TE/bin/transformers" ]; then
    echo ">>> fetching transformers @ $PINNED_SHA"
    rm -rf "$TFSRC"; mkdir -p "$TFSRC"; cd "$TFSRC"
    git init -q; git remote add origin https://github.com/huggingface/transformers
    git fetch -q --depth 1 origin "$PINNED_SHA"; git checkout -q FETCH_HEAD
    cd "$HERE"
    echo ">>> building venv + CPU torch + transformers + task deps"
    python3 -m venv "$TE"
    "$TE/bin/pip" install -q --no-cache-dir --upgrade pip
    "$TE/bin/pip" install -q --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
    "$TE/bin/pip" install -q --no-cache-dir "$TFSRC"
    "$TE/bin/pip" install -q --no-cache-dir timm pillow sentencepiece sacremoses librosa soundfile scipy accelerate protobuf openai
else
    echo ">>> toolenv present; skipping build"
fi

# Self-configuring bundle: a .pth points HF_HOME at the in-bundle hf-cache,
# resolved from the venv root (sys.prefix) so it holds wherever the bundle is
# mounted read-only. The agent invokes the bundle's python/transformers, so HF
# reads the prewarmed cache offline with no per-sandbox env setup. (Ubuntu venvs
# don't auto-import sitecustomize, hence the .pth + helper module.)
SP="$("$TE/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
cat > "$SP/_agentcap_hf_home.py" <<'PY'
import os
import sys

_cache = os.path.join(sys.prefix, "hf-cache")
if os.path.isdir(_cache):
    os.environ.setdefault("HF_HOME", _cache)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
PY
echo 'import _agentcap_hf_home' > "$SP/_agentcap_hf_home.pth"

echo ">>> sanity: CLI + pipeline import"
"$TE/bin/transformers" --help >/dev/null && echo "    transformers CLI OK"
"$TE/bin/python" -c "from transformers import pipeline" && echo "    pipeline import OK"

echo ">>> prewarming model cache (xet disabled)"
for m in $MODELS; do
    printf '    %-58s ' "$m"
    if "$TE/bin/python" - "$m" <<'PY' 2>/tmp/dl.err
import sys
from huggingface_hub import snapshot_download
# PyTorch + safetensors only; skip the TF/Flax/ONNX/Rust/GGUF weight copies
# transformers never loads (they triple the download for no benefit).
snapshot_download(sys.argv[1], ignore_patterns=[
    "*.h5", "tf_model*", "*.msgpack", "flax_model*", "*.onnx", "onnx/**",
    "*.tflite", "rust_model.ot", "*.gguf",
])
PY
    then echo "ok"; else echo "FAILED"; tail -2 /tmp/dl.err; fi
done
echo ">>> DONE. bundle at $TE ($(du -sh "$TE" | cut -f1))"
IN_CONTAINER
