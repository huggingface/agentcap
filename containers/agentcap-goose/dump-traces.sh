#!/bin/sh
# Post-corpus session dump for goose. Called once by the orchestrator
# after the run completes (``sandbox.run(["dump-traces"])``). Idempotent
# — re-running on the same set of sessions overwrites JSON files.
#
# Goose stores sessions in a SQLite store
# (/root/.local/share/goose/sessions/sessions.db); the per-session
# symlink-the-dir trick used by pi/hermes doesn't apply. Instead we
# enumerate sessions with ``goose session list --format json`` and
# call ``goose session export`` once per id.
set -e
[ -n "${AGENTCAP_TRACES_DIR:-}" ] && [ -d "$AGENTCAP_TRACES_DIR" ] || exit 0
# ``goose session list -f json`` emits an array of objects with at
# least ``id`` and ``name``. Filter to ids with python3 (jq is not
# in the image).
goose session list --format json 2>/dev/null \
    | python3 -c "
import json, sys
try:
    rows = json.load(sys.stdin)
except Exception:
    rows = []
for r in rows:
    sid = r.get('id') or r.get('session_id')
    if sid:
        print(sid)
" \
    | while read -r id; do
    [ -z "$id" ] && continue
    out="$AGENTCAP_TRACES_DIR/${id}.json"
    goose session export --session-id "$id" --format json -o "$out" 2>/dev/null \
        || echo "dump-traces: export failed for $id" >&2
done
