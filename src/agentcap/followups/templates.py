"""Rotating-template follow-up strategy.

Cycles through a small fixed pool. No extra inference cost; minor
variation in user-message tokens compared to plain ``continue``.
"""

from __future__ import annotations

from typing import Sequence

from . import FollowUp


_DEFAULT_POOL = ("continue", "go on", "what else?", "keep going")


class TemplatesFollowUp(FollowUp):
    name = "templates"

    def __init__(self, pool: Sequence[str] = _DEFAULT_POOL) -> None:
        if not pool:
            raise ValueError("templates pool must be non-empty")
        self.pool = list(pool)

    def next(self, *, original_task: str, last_response: str, turn: int) -> str:
        # turn=2 (first follow-up) → pool[0]; turn=3 → pool[1]; etc.
        idx = (turn - 2) % len(self.pool)
        return self.pool[idx]
