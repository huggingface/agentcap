# Contributing

agentcap is in early development — interfaces may shift while the
producer/consumer split settles. Issues, PRs, and capture-quality
reports against new (agent, model) tuples are all welcome.

## Setup

```bash
git clone https://github.com/huggingface/agentcap
cd agentcap
cargo build
```

The toolchain is pinned in [rust-toolchain.toml](rust-toolchain.toml) — rustup
fetches it automatically.

## Running tests

```bash
cargo test                            # unit + loopback proxy integration (what CI runs)
cargo test --test live -- --ignored   # live tier; skips if no model server (needs podman)
```

Live integration tests (real agent + model server inside podman) require local
prerequisites — see "Running tests" in the README.

Lint:

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
```

Both lint and unit tests must pass for CI to be green.

## Design decisions

Before touching the capture/export split or the on-disk file shapes,
read [AGENTS.md](AGENTS.md) — it lists the architecture decisions
that are settled and should be raised with maintainers before being
relitigated.

## Reporting capture quality

If you exercise a new `(agent, model)` tuple end-to-end, a PR
extending [docs/tested-models-and-agents.md](docs/tested-models-and-agents.md)
with the result is the most useful contribution. Include the model
quant (for GGUFs), the backend (llama.app / Inference Providers), and
any agent-specific gotchas that surfaced.

## Filing issues

When reporting a bug, include:

- The full `agentcap` command line.
- The capture dir's `run.json` (it has the agent, upstream, turn count).
- A single `.request.json` / `.response.json` pair from the failing
  turn if the issue is capture-related — the bodies are the source
  of truth.

## License

By contributing, you agree your contributions are licensed under the
Apache 2.0 License — see [LICENSE](LICENSE).
