# Tested models and agents

What has been verified end-to-end through agentcap. Each cell is one
observed (model, agent) tuple at a single point in time — re-run if
the model card or agent revs.

| model (Q4_K_M)                  | hermes | goose | pi | opencode |
|---|:-:|:-:|:-:|:-:|
| `Qwen3-Coder-30B-A3B-Instruct`  | ✅¹ | — | — | — |
| `Qwen3.6-35B-A3B`               | ✅ | ✅ | ✅ | ✅ |
| `unsloth/gemma-4-26B-A4B-it`    | ✅¹ | ✅¹ | ✅¹ | ✅¹ |
| `google/gemma-4-E4B-it`         | ✅¹ | ✅¹ | ✅¹ | ✅¹ |

`—` = not exercised.

1. Validated end-to-end through the full 30-prompt × 4-turn
   `examples/transformers-coding-session` corpus. Parquets live under
   the `transformers-coding-session/` prefix in
   `dacorvo/agentcap-captures` (private dataset). A few 26 B runs lost
   1 task to a 1200 s timeout; the rest of the corpus rendered
   cleanly.

## Operational notes

- Pre-agentic-era models do not work at any size — fail at tool
  emission.
- Reasoning-by-default models (Qwen 3.5+, 3.6) put their answer in
  `message.reasoning_content`, not `content`. Pass `--reasoning off`
  to `llama serve`.
- Hermes requires model context ≥ 64 K. Either raise `CTX_SIZE` or
  use `HermesDriver(context_length_override=65536)` to lie via the
  per-run config overlay (the user's `~/.hermes` is never touched).
- OpenCode hangs ≥ 30 min if launched from an empty directory
  (recursive glob from filesystem root). Always launch from a real
  project dir.
- VRAM estimates run ~2× low. On 4× A10G, 30–35 B Q4_K_M fits
  comfortably at 64 K; Q5_K_M and above eat into the per-card budget.
