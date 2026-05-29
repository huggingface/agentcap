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
#   N_GPU_LAYERS=999  TENSOR_SPLIT=1,1,1,1

set -euo pipefail

REPO="${1:?usage: $0 <hf-repo>[:<quant>]}"

command -v llama >/dev/null 2>&1 || curl -fsSL https://llama.app/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

exec llama serve \
    -hf "$REPO" \
    --alias "${REPO%:*}" \
    --host "${HOST:-0.0.0.0}" \
    --port "${PORT:-8000}" \
    --ctx-size "${CTX_SIZE:-32768}" \
    --reasoning "${REASONING:-auto}" \
    --n-gpu-layers "${N_GPU_LAYERS:-999}" \
    --tensor-split "${TENSOR_SPLIT:-1,1,1,1}" \
    --jinja
