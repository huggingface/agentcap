"""Literal-``continue`` follow-up strategy."""

from __future__ import annotations

from . import FollowUp


class ContinueFollowUp(FollowUp):
    name = "continue"

    def __init__(self, text: str = "continue") -> None:
        self.text = text

    def next(self, *, original_task: str, last_response: str, turn: int) -> str:
        return self.text
