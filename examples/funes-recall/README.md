# funes-recall

An agentcap corpus that drives a coding agent (pi, hermes, or opencode) through
**past-decision** questions — ones it can't answer from the files in front of it,
only by recalling earlier sessions. The agent is given funes (recall over past
AI-agent sessions) as a tool, and agentcap captures the whole agent ↔ model
exchange, so you can see *whether* a model reaches for memory and *how* it uses
what comes back. The sandbox cwd is empty: nothing to grep, so the only path to
the answer is recall.

## Run

```bash
./build-funesenv.sh                           # one-time: build the funes bundle

# with an OpenAI-compatible server on $UPSTREAM (default http://127.0.0.1:8001):
./run.sh --model GLM-4.5-Air                    # pi (default)
./run.sh --model GLM-4.5-Air --agent hermes
./run.sh --model GLM-4.5-Air --agent opencode
```

Captures land under `./.agentcap/` — `agentcap ls` to list,
`agentcap export … --push <owner>/<dataset>` to publish. The tasks are in `tasks.txt`;
`./run.sh --help` lists the env knobs.

## What a good run shows

The point is the cross-agent comparison: **an agent's tools set its options, and a
native decoy is where the model shows.** Example captures and traces are published at
[`dacorvo/funes-recall-session`](https://huggingface.co/collections/dacorvo/funes-recall-session)
(GLM-4.5-Air on Linux, alongside macOS/gemma-4-E4B):

| Agent | Model | How it reaches funes |
|---|---|---|
| **pi** | gemma-4-E4B | first action: `recall` → `get` |
| **pi** | GLM-4.5-Air | first action: `recall` → `get` |
| **opencode** | gemma-4-E4B | first action: `funes_recall` → `funes_get` |
| **opencode** | GLM-4.5-Air | first action: `funes_recall` → `funes_get` |
| **hermes** | gemma-4-E4B | straight to `mcp_funes_recall` |
| **hermes** | GLM-4.5-Air | hits its native `session_search` 10–18× first, then funes |

Each task's answer lives in the corpus — e.g. front-loading **0.1–0.3% procedural
data** beats standard pretraining at 55–86% of the training data.

## Notes

- `FUNES_REMOTE` (default `dacorvo/funes-bench`, public) is a **synthetic** corpus —
  Fable-5 traces of a toy transformer project, not funes' own history. Point it elsewhere
  and the tasks have to match what that index records; a **private** remote needs a
  read-only HF token (`HF_TOKEN`, else `~/.cache/huggingface/token`).
- **hermes needs a ≥64K-context server** (e.g. `llama-server -c 65536`); its
  exploration is verbose and a turn otherwise dies with `exceed_context_size`. pi runs
  fit in 32K.
- Needs a **funes** with `funes install` for pi, hermes, and opencode (the bundle pulls
  the latest binary from the HF bucket).
