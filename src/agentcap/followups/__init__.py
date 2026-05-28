"""Follow-up strategies for multi-turn agent runs.

Each strategy implements ``FollowUp.next(...)`` returning the next
user message to feed to the agent given the prior turn's response and
the original task. Strategies are stateful (``templates`` rotates a
pool, ``synthesized`` may keep a model client) but the contract is
the same.

Three built-in strategies, in increasing order of cost / realism:

  - ``continue`` (default): the literal string ``"continue"``. Cheapest
    and maximises cross-session match opportunity since user-message
    tokens are byte-identical across sessions.
  - ``templates``: rotates through a small pool (``"continue"``,
    ``"go on"``, ``"what else?"``, ``"keep going"``).
  - ``synthesized``: feeds (original task + agent's last response)
    into a separate model call to produce a realistic follow-up. The
    synthesizer call **bypasses the capture proxy** by design — its
    requests are not part of the capture.
"""

from __future__ import annotations

import abc


class FollowUp(abc.ABC):
    """Strategy for picking the next user message in a multi-turn run."""

    name: str

    @abc.abstractmethod
    def next(self, *, original_task: str, last_response: str, turn: int) -> str:
        """Return the next user message.

        ``turn`` is the 1-indexed number of the *upcoming* turn (so the
        first follow-up is ``turn=2`` because the original task was
        turn 1). Strategies that don't care about ``turn`` simply ignore
        the arg.
        """


def get_followup(name: str, **kwargs) -> FollowUp:
    if name == "continue":
        from .continue_ import ContinueFollowUp

        return ContinueFollowUp(**kwargs)
    if name == "templates":
        from .templates import TemplatesFollowUp

        return TemplatesFollowUp(**kwargs)
    if name == "synthesized":
        from .synthesized import SynthesizedFollowUp

        return SynthesizedFollowUp(**kwargs)
    raise ValueError(f"unknown follow-up strategy: {name!r}")


__all__ = ["FollowUp", "get_followup"]
