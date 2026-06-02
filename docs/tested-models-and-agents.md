# Tested models and agents

What has been verified end-to-end through agentcap. Each cell is one
observed (model, agent) tuple at a single point in time — re-run if
the model card or agent revs.

| model (Q4_K_M)                 | hermes | goose | pi | opencode |
|---|:-:|:-:|:-:|:-:|
| `gemma-4-E4B-it`               | ✅ | ✅ | ✅ | ✅ |
| `gemma-4-26B-A4B-it`           | ✅ | ✅ | ✅ | ✅ |
| `Qwen3.6-35B-A3B`              | ✅ | ✅ | ✅ | ✅ |
| `Qwen3-Coder-30B-A3B-Instruct` | ✅ | ✅ | ✅ | ✅ |

`—` = not exercised. Captures live in the
[`transformers-coding-session`](https://huggingface.co/collections/dacorvo/transformers-coding-session-6a1de25f14ed2323176e6c39)
and
[`hf-hub-session`](https://huggingface.co/collections/dacorvo/hf-hub-session-6a1dd66a68425ef2b98ee273)
Hub Collections.

## Agent-specific notes

- Pre-agentic-era models do not work at any size — fail at tool
  emission.
- OpenCode hangs ≥ 30 min if launched from an empty directory
  (recursive glob from filesystem root). Always launch from a real
  project dir.

For server-launch recipes (REASONING / CTX_SIZE / VRAM layout per
model), see [scripts/MODELS.md](../scripts/MODELS.md).
