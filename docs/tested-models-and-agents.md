# Tested models and agents

What has been verified end-to-end through agentcap. Each cell is one
observed (model, agent) tuple at a single point in time — re-run if
the model card or agent revs.

| model (Q4_K_M)                  | Hermes | Goose | pi-mono | OpenCode |
|---|:-:|:-:|:-:|:-:|
| `Qwen3-Coder-30B-A3B-Instruct`  | ✅ | — | — | — |
| `Qwen3.6-35B-A3B`               | ✅ | ✅ | ✅ | ✅ |
| `unsloth/gemma-4-26B-A4B-it`    | ❌¹ | ✅ | ✅ | ✅ |
| `Qwen3-4B-Instruct-2507` (CPU)  | ✅² | ✅ | ✅ | ✅² |
| `google/gemma-4-E4B-it`         | ❌¹ | — | — | — |

`—` = not exercised. The "validated" bar is the docstring task (live
tests) or, for the Gemma-26B row, a single planning task; none of
these have been driven through the full 30-prompt corpus yet.

1. ❌ = does not load Hermes' skill machinery. The Gemma family
   doesn't pick up Hermes' "MUST load a skill" directive regardless
   of size; the model can still tool-call, just not run the Hermes
   loop end-to-end.
2. Hermes and OpenCode require live-test trims to pass on a 4 K-ctx
   CPU server:
   `HermesDriver(ignore_rules=True, toolsets="file", context_length_override=65536)`
   and `OpenCodeDriver(minimal_agent=True)`. Off in production.

## Operational notes

- Pre-agentic-era models do not work at any size — fail at tool
  emission.
- Reasoning-by-default models (Qwen 3.5+, 3.6) put their answer in
  `message.reasoning_content`, not `content`. Pass `--reasoning off`
  to llama-server.
- Hermes requires model context ≥ 64 K. Either raise `CTX_SIZE` or
  use `HermesDriver(context_length_override=65536)` to lie via the
  per-run config overlay (the user's `~/.hermes` is never touched).
- OpenCode hangs ≥ 30 min if launched from an empty directory
  (recursive glob from filesystem root). Always launch from a real
  project dir.
- VRAM estimates run ~2× low. On 4× A10G, 30–35 B Q4_K_M fits
  comfortably at 64 K; Q5_K_M and above eat into the per-card budget.
