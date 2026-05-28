# transformers-coding-session

Exercises the **generic tools** every coding agent ships with —
file read/write/edit, grep, bash, web fetch — against a real,
non-trivial codebase: [`huggingface/transformers`](https://github.com/huggingface/transformers).

The prompts are open-ended planning / diagnostic questions about
the transformers source tree; multi-turn follow-ups push the agent
into the code. See [tasks.txt](tasks.txt) for the corpus and how
it was drafted.

## Prereqs

A local clone of `huggingface/transformers`. `run.sh` seeds
`<WORKDIR>/sandbox/` as a detached `git worktree` of it so the
agent has real code to inspect:

```bash
git clone https://github.com/huggingface/transformers ~/transformers
```

`run.sh` auto-detects `~/transformers` or `./transformers`;
otherwise set `TRANSFORMERS_CHECKOUT=<path>`.

## Run

```bash
./run.sh --agent hermes --model google/gemma-4-E4B-it
./export.sh                    # latest workdir → corpus bucket
```

`./run.sh --help` and `./export.sh --help` for the env-var knobs.
