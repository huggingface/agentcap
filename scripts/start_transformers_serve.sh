#!/usr/bin/env bash
# Launch `transformers serve` for Gemma-4 across all 4 A10G GPUs.
#
# Why this script exists:
#   1. The agentcap repo's own .venv has no torch/CUDA. We reuse the
#      parent project's venv (/home/ubuntu/hf-mount-cache-examples/.venv)
#      which has torch 2.10 + CUDA 12.8 already set up.
#   2. We override the editable transformers install via PYTHONPATH so
#      the model server picks up the *user's* /home/ubuntu/transformers
#      checkout. That checkout must be on branch `agentcap-sdpa-fix`,
#      which carries the SDPA query-dim chunking patch.
#   3. The chunking patch is required for long-context Gemma-4 inference
#      on this stack:
#        - Gemma-4 is registered as multimodal in MODEL_FOR_CAUSAL_LM_-
#          MAPPING_NAMES, so transformers serve loads
#          Gemma4ForConditionalGeneration regardless of --task hint.
#        - Modality detection then returns VLM, so --continuous-batching
#          short-circuits to sequential generate(), which means
#          single-shot prefill (no upstream chunked prefill).
#        - Single-shot prefill on Gemma-4's hybrid sliding-window mask
#          forces SDPA into the math fallback (FA / mem-efficient
#          backends can't handle that mask shape), which materialises
#          a seq²·heads·fp32 attention-scores tensor and OOMs around
#          ~10k tokens on A10G.
#        - The patch chunks SDPA along the query dim (default 4096
#          tokens), bounding peak attention-scores memory. Bit-
#          equivalent to the unchunked call.
#      A proper upstream fix would either (a) register a Gemma-4
#      LM-only auto-class with prefix-stripping checkpoint load, or
#      (b) install flash-attn 2 (FA2 supports sliding masks).
#
# Defaults:
#   MODEL=google/gemma-4-E4B-it    -- already in ~/.cache/huggingface
#   HOST=0.0.0.0  PORT=8000
#   TSERVE_SDPA_CHUNK_Q=4096       -- 0 disables chunking
#
# Usage:  ./scripts/start_transformers_serve.sh
#         (Ctrl-C to stop. Cold load ~2 min on 4× A10G; warm load ~2 s.)

set -euo pipefail

MODEL="${MODEL:-google/gemma-4-E4B-it}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TRANSFORMERS_SRC="${TRANSFORMERS_SRC:-/home/ubuntu/transformers/src}"
VENV="${VENV:-/home/ubuntu/hf-mount-cache-examples/.venv}"

if [ ! -d "$TRANSFORMERS_SRC" ]; then
    echo "error: TRANSFORMERS_SRC=$TRANSFORMERS_SRC does not exist" >&2
    exit 1
fi
if [ ! -x "$VENV/bin/transformers" ]; then
    echo "error: $VENV/bin/transformers not found — wrong venv?" >&2
    exit 1
fi
if ! grep -q '_CHUNK_Q_THRESHOLD' \
        "$TRANSFORMERS_SRC/transformers/integrations/sdpa_attention.py"; then
    echo "error: SDPA chunking patch missing from $TRANSFORMERS_SRC." >&2
    echo "       cd $(dirname $TRANSFORMERS_SRC) && git checkout agentcap-sdpa-fix" >&2
    exit 1
fi

echo "[start] model=$MODEL listen=$HOST:$PORT"
echo "[start] transformers src=$TRANSFORMERS_SRC venv=$VENV"
echo "[start] sdpa chunk_q=${TSERVE_SDPA_CHUNK_Q:-4096}"

export PYTHONPATH="$TRANSFORMERS_SRC${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export TSERVE_SDPA_CHUNK_Q="${TSERVE_SDPA_CHUNK_Q:-4096}"

exec "$VENV/bin/transformers" serve \
    --device balanced \
    --dtype bfloat16 \
    --host "$HOST" \
    --port "$PORT" \
    "$MODEL"
