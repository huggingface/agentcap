# Model launch recipes

Reference for the models the published bucket runs used. All recipes
invoke `scripts/start_llama_cpp_server.sh`, which downloads the GGUF
on first use and runs `llama serve`.

## Model facts

These apply on any host, regardless of GPU count.

| Model | GGUF repo | Reasoning |
|---|---|---|
| gemma-4-E4B-it | `ggml-org/gemma-4-E4B-it-GGUF` | `auto` |
| gemma-4-26B-A4B-it | `ggml-org/gemma-4-26B-A4B-it-GGUF` | `auto` |
| Qwen3-Coder-30B-A3B-Instruct | `unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF`* | `auto` |
| Qwen3.6-35B-A3B | `ggml-org/Qwen3.6-35B-A3B-GGUF` | `off` |
| GLM-4.5-Air | `unsloth/GLM-4.5-Air-GGUF` | `off` |

*`ggml-org` doesn't publish Qwen3-Coder; `unsloth` is the canonical
source.

Two rules that bite regardless of host:

- **`REASONING=off`** is needed whenever the model answers in
  `reasoning_content` rather than `content` (Qwen 3.5+, 3.6, GLM
  thinking-mode). Hermes' tool-call parser only looks at `content`,
  so leaving thinking on breaks the agent's tool-use loop.
  Qwen3-Coder is not reasoning-by-default, so `auto` is fine.

- **`CTX_SIZE=65536`** is needed for the 30-task corpora. Hermes
  refuses to start unless both `model.context_length` and
  `auxiliary.compression.context_length` ≥ 64 K, and we bake both
  at 65536; the `llama serve` side must match. The script's default
  is 32768, which is fine for short smoke runs but not for the full
  corpora.

- **`FIT=off`** is the script's default. Auto-fit hits
  `GGML_SCHED_MAX_SPLIT_INPUTS` on multi-GPU for several of these
  models; the simpler scheduler avoids it.

For the `agentcap run` side, pass the GGUF repo id as `--model` so
the captured `request.model` matches the alias the server advertises
and the export's tokenizer auto-detect lines up.

---

## Sample configuration: 4× A10G

The dev host used to build the bucket has 4× A10G (22 GB each). The
commands below pin GPU count via `TENSOR_SPLIT`; on a different
layout (one bigger GPU, fewer GPUs, etc.), omit `TENSOR_SPLIT` to
let llama auto-detect across whatever's visible.

```bash
# gemma-4-E4B-it — fits on one A10G at Q4_K_M
CTX_SIZE=65536 \
    scripts/start_llama_cpp_server.sh ggml-org/gemma-4-E4B-it-GGUF

# gemma-4-26B-A4B-it — ~16 GB at Q4_K_M; 1 A10G is tight with KV
CTX_SIZE=65536 TENSOR_SPLIT=1,1 \
    scripts/start_llama_cpp_server.sh ggml-org/gemma-4-26B-A4B-it-GGUF

# Qwen3-Coder-30B-A3B-Instruct — 4 GPUs (2 also works on bigger cards)
CTX_SIZE=65536 TENSOR_SPLIT=1,1,1,1 \
    scripts/start_llama_cpp_server.sh unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF

# Qwen3.6-35B-A3B — needs 4× A10G; suppress visible reasoning
CTX_SIZE=65536 TENSOR_SPLIT=1,1,1,1 REASONING=off \
    scripts/start_llama_cpp_server.sh ggml-org/Qwen3.6-35B-A3B-GGUF

# GLM-4.5-Air — 4 GPUs, suppress visible reasoning
CTX_SIZE=65536 TENSOR_SPLIT=1,1,1,1 REASONING=off \
    scripts/start_llama_cpp_server.sh unsloth/GLM-4.5-Air-GGUF
```
