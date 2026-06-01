#!/bin/sh
# Post-corpus session dump for opencode. Called once by the
# orchestrator after the run completes
# (``sandbox.run(["dump-traces"])``). Idempotent.
#
# Opencode stores sessions in a SQLite store; the per-session
# symlink-the-dir trick used by pi/hermes doesn't apply. Instead we
# enumerate sessions with ``opencode session list --format json``
# and call ``opencode export <sessionID>`` once per id.
set -e
[ -n "${AGENTCAP_TRACES_DIR:-}" ] && [ -d "$AGENTCAP_TRACES_DIR" ] || exit 0
opencode session list --format json 2>/dev/null \
    | python3 -c "
import json, sys
try:
    rows = json.load(sys.stdin)
except Exception:
    rows = []
for r in rows:
    sid = r.get('id') or r.get('sessionID')
    if sid:
        print(sid)
" \
    | while read -r id; do
    [ -z "$id" ] && continue
    out="$AGENTCAP_TRACES_DIR/${id}.json"
    opencode export "$id" 2>/dev/null > "$out" \
        || echo "dump-traces: export failed for $id" >&2
done
