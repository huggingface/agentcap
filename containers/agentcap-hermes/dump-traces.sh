#!/bin/sh
# Post-corpus session dump for hermes. Called once by the orchestrator
# after the run completes (``sandbox.run(["dump-traces"])``). Idempotent.
#
# Hermes stores sessions in a SQLite state.db; per-session JSONL files
# don't exist on disk. ``hermes sessions export`` dumps the full set
# to one JSONL — one line per session (or session-event, depending on
# the hermes version). The single-file shape is fine for the traces
# dataset: downstream readers can group by session_id.
set -e
[ -n "${AGENTCAP_TRACES_DIR:-}" ] && [ -d "$AGENTCAP_TRACES_DIR" ] || exit 0
hermes sessions export "$AGENTCAP_TRACES_DIR/sessions.jsonl" 2>/dev/null \
    || echo "dump-traces: hermes sessions export failed" >&2
