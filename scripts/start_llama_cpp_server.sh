#!/usr/bin/env bash
# Launch llama.cpp's server against an HF GGUF repo.
# Auto-installs llama (https://llama.app) on first use.
#
# Usage:
#   ./start_llama_cpp_server.sh <hf-repo>[:<quant>]
#
# Example:
#   ./start_llama_cpp_server.sh ggml-org/gemma-4-E4B-it-GGUF
#   ./start_llama_cpp_server.sh unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF
#
# Prefer ggml-org/ repos (the llama.cpp team's canonical quants);
# fall back to unsloth/ for models ggml-org hasn't published.
#
# Env (sensible defaults):
#   HOST=0.0.0.0  PORT=8000  CTX_SIZE=32768  REASONING=auto
#   FIT=off  (auto-fit hits GGML_SCHED_MAX_SPLIT_INPUTS on multi-GPU
#            for some models; ``on`` re-enables it)
#   N_GPU_LAYERS  default: llama's own ``auto``. Set to ``all`` /
#                 ``999`` to force all-to-VRAM.
#   TENSOR_SPLIT  default: llama splits evenly across visible GPUs.
#                 Set e.g. ``1,1,1,1`` to override.

set -euo pipefail

REPO="${1:?usage: $0 <hf-repo>[:<quant>]}"

command -v llama >/dev/null 2>&1 || curl -fsSL https://llama.app/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

opt=()
[ -n "${N_GPU_LAYERS:-}" ] && opt+=(--n-gpu-layers "$N_GPU_LAYERS")
[ -n "${TENSOR_SPLIT:-}" ] && opt+=(--tensor-split "$TENSOR_SPLIT")

exec llama serve \
    -hf "$REPO" \
    --alias "${REPO%:*}" \
    --host "${HOST:-0.0.0.0}" \
    --port "${PORT:-8000}" \
    --ctx-size "${CTX_SIZE:-32768}" \
    --reasoning "${REASONING:-auto}" \
    --fit "${FIT:-off}" \
    ${opt[@]+"${opt[@]}"} \
    --jinja
