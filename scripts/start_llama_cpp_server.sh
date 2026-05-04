#!/usr/bin/env bash
# Launch llama.cpp's `llama-server` against a GGUF, across all 4 A10G GPUs.
#
# Prereqs (one-time):
#   1. Build llama.cpp with CUDA + server enabled:
#        export PATH="/home/ubuntu/agentcap/.venv/bin:/usr/local/cuda/bin:$PATH"
#        cd /home/ubuntu/llama.cpp
#        cmake -B build -DGGML_CUDA=ON -DLLAMA_BUILD_SERVER=ON
#        cmake --build build --config Release -j --target llama-server
#   2. Obtain a GGUF (e.g. Q4_K_M from unsloth) and pass it via GGUF_PATH:
#        hf download unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF \
#            --include "Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf" \
#            --local-dir /home/ubuntu/llama.cpp/models/qwen3-coder-30b-a3b
#
# Required env:
#   GGUF_PATH         -- model file
# Optional env (sensible defaults):
#   MODEL_ALIAS       -- name advertised on /v1/models. Defaults to the
#                        GGUF's basename minus ".gguf".
#   HOST=0.0.0.0  PORT=8000
#   N_GPU_LAYERS=999  -- offload all layers
#   TENSOR_SPLIT=1,1,1,1 -- equal split across 4 GPUs
#   CTX_SIZE=32768    -- KV cache size in tokens
#   REASONING=auto    -- "auto" follows the model's chat-template default;
#                        "off" suppresses thinking blocks. Set "off" for
#                        reason-by-default models (Qwen3.5+ etc.) that put
#                        their visible answer in `reasoning_content` rather
#                        than `content` — that breaks Hermes, which parses
#                        tool calls from `content`.

set -euo pipefail

LLAMA_DIR="${LLAMA_DIR:-/home/ubuntu/llama.cpp}"
LLAMA_SERVER="$LLAMA_DIR/build/bin/llama-server"
GGUF_PATH="${GGUF_PATH:-}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
N_GPU_LAYERS="${N_GPU_LAYERS:-999}"
TENSOR_SPLIT="${TENSOR_SPLIT:-1,1,1,1}"
CTX_SIZE="${CTX_SIZE:-32768}"
REASONING="${REASONING:-auto}"

if [ ! -x "$LLAMA_SERVER" ]; then
    echo "error: $LLAMA_SERVER not found — build it (see header)." >&2
    exit 1
fi

if [ -z "$GGUF_PATH" ] || [ ! -f "$GGUF_PATH" ]; then
    echo "error: GGUF_PATH not set or file missing." >&2
    echo "       Example: GGUF_PATH=$LLAMA_DIR/models/qwen3-coder-30b-a3b/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf $0" >&2
    exit 1
fi

# Default the alias to the GGUF basename so /v1/models has something
# meaningful and so the agentcap captures' `model` field is sensible.
if [ -z "${MODEL_ALIAS:-}" ]; then
    MODEL_ALIAS=$(basename "$GGUF_PATH" .gguf)
fi

echo "[start] gguf=$GGUF_PATH listen=$HOST:$PORT alias=$MODEL_ALIAS"
echo "[start] gpu_layers=$N_GPU_LAYERS tensor_split=$TENSOR_SPLIT ctx_size=$CTX_SIZE reasoning=$REASONING"

exec "$LLAMA_SERVER" \
    --model "$GGUF_PATH" \
    --alias "$MODEL_ALIAS" \
    --host "$HOST" \
    --port "$PORT" \
    --n-gpu-layers "$N_GPU_LAYERS" \
    --tensor-split "$TENSOR_SPLIT" \
    --ctx-size "$CTX_SIZE" \
    --reasoning "$REASONING" \
    --jinja
