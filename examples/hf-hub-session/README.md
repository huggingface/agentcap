# hf-hub-session

Exercises the [huggingface/skills](https://github.com/huggingface/skills)
bundle — the agent reaches for `hf` CLI commands, the
`huggingface_hub` SDK, and direct HTTPS calls against
`https://huggingface.co/api/...` to answer Hub-related questions
(uploading, storage, Spaces, Collections, …).

The sandbox starts empty by design — there's no codebase to grep,
only the Hub. Without the skills bundle the agent has no grounding
and the loop ends in "skill not found." See [tasks.txt](tasks.txt)
for the corpus and how it was drafted.

## Prereqs

A local clone of `huggingface/skills`. `run.sh` bind-mounts it
read-only into the sandbox and wires it into the agent's discovery
path:

```bash
git clone https://github.com/huggingface/skills ~/skills
```

`run.sh` auto-detects `~/skills`, `~/dev/skills`, or `./skills`;
otherwise set `SKILLS_CHECKOUT=<path>` (or let `run.sh` fetch it
into `~/.cache/agentcap/hf-skills` on first use).

If the agent needs Hub credentials (private repos, write access),
either run `hf auth login` on the host first or export `HF_TOKEN`.

## Run

```bash
./run.sh --agent hermes --model google/gemma-4-E4B-it
./export.sh                    # latest workdir → corpus dataset
```

`./run.sh --help` and `./export.sh --help` for the env-var knobs.

Both scripts pin `AGENTCAP_WORKSPACE` to this directory, so runs
live under `./.agentcap/` rather than the global workspace. List
them with `agentcap ls` *from this directory* (or
`AGENTCAP_WORKSPACE=$PWD agentcap ls` from anywhere).
