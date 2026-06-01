#!/usr/bin/env python3
"""One-off backfill of ``task_id`` + ``turn`` onto pre-stamping captures.

Captures produced before the proxy started stamping orchestrator context
lack the ``task_id`` and ``turn`` fields. This script reconstructs them
by grouping captures by their last user message (which is constant
inside one turn but distinct across turns when followups are synthesized)
and matching the chronological group order against ``run.json``'s task /
turn structure.

This is throwaway — fresh runs against an instrumented proxy will
produce self-describing captures, and the agent-side traces these old
captures are missing have to come from a rerun anyway.

Usage:
    python scripts/backfill_capture_metadata.py <run-dir> [--write]

Without ``--write`` the script prints what it would do; with ``--write``
it rewrites each ``<rid>.request.json`` in place with the two extra
fields.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _last_user(messages: list) -> str | None:
    for m in reversed(messages or []):
        if m.get("role") != "user":
            continue
        content = m.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                p.get("text", "") for p in content if isinstance(p, dict)
            )
    return None


def backfill(run_dir: Path, write: bool) -> int:
    cap_dir = run_dir / "captures"
    run_json_path = run_dir / "run.json"
    if not cap_dir.is_dir() or not run_json_path.is_file():
        print(f"error: {run_dir} is not an agentcap run dir", file=sys.stderr)
        return 2

    run = json.loads(run_json_path.read_text())
    tasks = run.get("tasks") or []
    task_prompts = [(t.get("task_id") or "?", t.get("prompt") or "") for t in tasks]

    captures: list[dict] = []
    for path in sorted(cap_dir.glob("*.request.json")):
        try:
            rec = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        body = rec.get("body") or {}
        captures.append({
            "path": path,
            "rec": rec,
            "captured_at": int(rec.get("captured_at", 0)),
            "last_user": _last_user(body.get("messages") or []),
        })
    captures.sort(key=lambda c: c["captured_at"])

    print(f"run: {run_dir.name}")
    print(f"  captures: {len(captures)}")
    print(f"  tasks declared: {len(task_prompts)}")
    if not captures or not task_prompts:
        return 1

    # Anchor: assign each capture to the task whose initial prompt last
    # appeared (in chronological order). The first capture whose
    # last_user == tasks[i].prompt opens task i's window; everything
    # between i's anchor and i+1's anchor belongs to task i.
    assignment: list[tuple[str, int] | None] = [None] * len(captures)
    current_task: int = -1
    current_turn: int = 0  # 0 = uninitialised
    current_followup: str | None = None
    anchor_misses = 0
    for i, c in enumerate(captures):
        # Does this capture match the next task's initial prompt? If so
        # we've crossed a task boundary.
        next_task = current_task + 1
        if next_task < len(task_prompts) and c["last_user"] == task_prompts[next_task][1]:
            current_task = next_task
            current_turn = 1
            current_followup = None
        elif current_task < 0:
            # Captures before any anchor — likely first turn 1 missing
            # match (rare). Snap to task 0 turn 1.
            current_task = 0
            current_turn = 1
            anchor_misses += 1
        elif c["last_user"] == task_prompts[current_task][1]:
            # Still on turn 1 of the same task.
            current_turn = 1
        else:
            # Different last_user from the task's initial prompt and from
            # any task anchor — either a follow-up or a hermes repair
            # message. We treat the first such transition as turn 2 and
            # collapse subsequent variants under the same turn.
            if current_followup is None or current_followup == c["last_user"]:
                if current_turn < 2:
                    current_turn = 2
                current_followup = c["last_user"]
            else:
                # Different from the recorded followup — likely a repair
                # message; keep current_turn unchanged.
                pass
        assignment[i] = (task_prompts[current_task][0], current_turn)

    if anchor_misses:
        print(f"  ! {anchor_misses} captures arrived before any task anchor; check ordering")

    counts: dict[tuple[str, int], int] = {}
    for asg in assignment:
        counts[asg] = counts.get(asg, 0) + 1
    print("  bucket counts:")
    for (tid, turn), n in sorted(counts.items()):
        print(f"    {tid} turn={turn}: {n} captures")

    if not write:
        print("  (dry-run; pass --write to apply)")
        return 0

    n_written = 0
    for c, (tid, turn) in zip(captures, assignment):
        c["rec"]["task_id"] = tid
        c["rec"]["turn"] = turn
        c["path"].write_text(json.dumps(c["rec"], indent=2))
        n_written += 1
    print(f"  wrote {n_written} updated request.json files")
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)
    return backfill(args.run_dir, args.write)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
