# Containerfile owns the whole install

Status: **proposal**. Not implemented. The non-trivial follow-up.

## The actual problem

Today an agent's install is split between two phases:

* **Build-time (Containerfile)** — installs the agent binary,
  bootstraps a baseline config dir (`~/.hermes/`, etc.).
* **Run-time (driver Python)** — mints a per-run overlay, copies the
  baseline config into it, rewrites the proxy `base_url` /
  `context_length` / model into the copy, wipes per-run state dirs,
  sets `HERMES_HOME` / `XDG_CONFIG_HOME` / `PI_CODING_AGENT_DIR`
  pointing at the overlay.

The runtime side is install-like work — assembling config files the
agent reads at startup. It exists because three values vary across
agentcap-runs and can't be baked at build time:

| Value | Source |
|---|---|
| proxy `base_url` | depends on `agentcap run --listen` |
| `model` name | depends on `--model` |
| per-run state | needs fresh dirs (memories/, sessions/) per run |

If we can eliminate those reasons, the driver collapses to "build
argv, exec, parse." All install lives in the Containerfile.

## Proposal

Three changes, in order:

### 1. Fix the proxy URL to a constant

Drop `--listen` as a user-facing CLI flag; agentcap's in-process
capture proxy always binds to a known port (e.g. `127.0.0.1:8081`).
The agent always reaches the proxy at the same URL.

Same-effort change for Lima: bake `host.lima.internal:8081` into the
Lima provisioning scripts.

**Diff**: drop `--listen` and `_parse_listen` from
[src/agentcap/__main__.py](../../src/agentcap/__main__.py); define
the constant in [src/agentcap/proxy.py](../../src/agentcap/proxy.py).
~15 LoC.

### 2. Bake the full config into each Containerfile

Today's hermes Containerfile already does `hermes config set
model.provider custom` etc. to bootstrap a usable shell of
`config.yaml`. Push that further — include the proxy `base_url`,
`context_length`, and any other values that don't vary per run:

```dockerfile
RUN hermes --version >/dev/null \
 && hermes config set model.provider custom \
 && hermes config set model.base_url http://127.0.0.1:8081/v1 \
 && hermes config set model.context_length 65536 \
 && hermes config set auxiliary.compression.context_length 65536
```

Per-agent specifics:

* **hermes** — config.yaml baked. Model selected at runtime via the
  `-m MODEL` CLI flag (hermes accepts it; overrides the config).
* **goose** — env-only agent; bake `OPENAI_HOST` / `GOOSE_PROVIDER`
  defaults via `ENV` lines. Model passed per-run via `GOOSE_MODEL`.
* **opencode** — bake `~/.config/opencode/opencode.json` (opencode
  reads it from there, not just cwd). Model via `--model`.
* **pi** — bake `models.json` at a path the `PI_CODING_AGENT_DIR`
  env var (also baked) points to. Model via `--model`.

**Diff**: 5–10 lines per Containerfile. Four files.

### 3. Persistent buildah container per agentcap-run

Today each `sandbox.run()` opens and tears down a fresh `buildah
from` working container. That means each turn starts with the
image's pristine `/root/.hermes/` — agent state (memories, session
files) doesn't survive between turns. The current driver overlays
exist partly to provide that continuity (HERMES_HOME=overlay; the
agent writes there across multiple turns).

If `BwrapSandbox` instead owns one persistent buildah container for
the lifetime of an `agentcap run`:

* Turn 1 writes to `/root/.hermes/memories/`. That write goes to the
  buildah container's OverlayFS upper layer.
* Turn 2 bwraps into the *same* container; the previous turn's
  writes are visible — session state is naturally continuous.
* End of `agentcap run`: `buildah rm <container>` discards the
  upper layer. Per-run state never escapes the run.

No driver-side overlay needed. The Lima backend already works this
way (one VM lives across all turns).

**Diff**: [src/agentcap/sandbox/bwrap.py](../../src/agentcap/sandbox/bwrap.py)
gains an `__init__`/`close` pair that calls `buildah from` /
`buildah rm`; the inner shell script in `wrap()` references the
stored container id instead of `buildah from`-ing fresh each time.
~40 LoC.

## What drivers shrink to

Hermes today is ~290 LoC (driver + parsers). With the above, it's
~50 LoC: build argv, hand off to sandbox, parse output. No
`_ensure_overlay`, no `_rewrite_config`, no `_HERMES_FRESH_PER_RUN`
set, no `cp -aL`, no mkdir/rm bookkeeping. Goose, opencode, pi
shrink similarly (they don't have hermes's identity-snapshot dance,
so the savings are smaller in absolute terms — but the
"install-like work in the driver" goes away for all four).

## Trade-offs to call out

1. **`--listen` is gone.** Users who today set a non-default proxy
   port lose the option. Survey: probably zero real users of that
   flag. Worth dropping.
2. **Per-run state is ephemeral by buildah-container teardown.**
   Today the driver's overlay cleanup is explicit
   (`self.sandbox.rmtree(overlay)`); the new model relies on
   `buildah rm` doing the same job. Same end-state, less code.
3. **Containerfile complexity goes up slightly.** A few `RUN hermes
   config set …` lines that previously lived in driver Python. Worth
   it: one place, one phase.
4. **Bwrap sandbox gains lifecycle.** It becomes a context manager
   (or has explicit `close()`). `agentcap.__main__` owns the
   close, mirrors the Lima pattern.

## Migration

Order the work so each step lands independently and runs green:

1. Hardcode the proxy port; remove `--listen`; live tests pass with
   the constant.
2. Refactor `BwrapSandbox` to own a persistent container with
   `__init__` / `close`. Drivers still build overlays — no behavior
   change yet. Live tests pass.
3. Per-agent: move config writes from driver into Containerfile,
   delete the corresponding overlay code from the driver. One
   driver at a time; live test for that agent must pass before
   moving to the next.

Each step is its own commit, bisect-friendly.

## Why this is different from the previous proposal

The previous draft proposed a YAML spec + generic driver + parser
registry — three new layers of indirection to avoid four small
bespoke files. That was a maintenance loss disguised as a win.

This proposal adds *zero* new abstractions. It just shifts where
install lives (build-time, not run-time) and gives bwrap the same
single-container-per-run lifecycle Lima already has. The four
driver files stay; they just become smaller.
