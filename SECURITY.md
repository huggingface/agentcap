# Security Policy

## Reporting a vulnerability

If you find a security issue in agentcap, please report it privately
via [GitHub Security Advisories](https://github.com/huggingface/agentcap/security/advisories/new)
rather than opening a public issue.

Please include:

- The agentcap version (or commit SHA).
- A description of the vulnerability and its impact.
- Reproduction steps.

## Scope

agentcap is a developer tool: a local HTTP proxy plus an agent
orchestrator that bind-mounts host paths into a sandbox. Issues we
care about include:

- **Proxy**: request smuggling, header injection, capture bypass.
- **Sandbox**: agent escape from the bwrap/Lima boundary into the
  host filesystem or network.
- **Export**: credential leakage into captured `.request.json` /
  `.response.json` files or pushed parquets. (Authorization headers
  are deliberately not persisted; report any path that breaks this.)

## Out of scope

- The agent CLIs themselves (`hermes`, `goose`, `opencode`, `pi`)
  and their model backends — report upstream.
- Issues that require write access to the user's `~/.cache/huggingface`
  or their HF tokens — the user trusts those by configuring HF.
