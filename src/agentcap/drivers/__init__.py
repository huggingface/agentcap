"""Agent driver adapters.

A driver wraps an agent CLI (Hermes, OpenCode, …) so the orchestrator
can:

  - start a new session with an initial prompt,
  - resume an existing session for a follow-up prompt,
  - extract the final response text from each turn (for the
    follow-up synthesizer).

Drivers shell out to the agent's binary; they do not implement the
agent's semantics. Configuring the agent to point at the capture proxy
(via config file or env) is the orchestrator's responsibility.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Callable


@dataclass
class AgentTurn:
    """One turn of agent execution."""

    session_id: str | None
    response_text: str
    returncode: int
    stdout: str
    stderr: str


class AgentDriver(abc.ABC):
    """Abstract adapter wrapping an agent CLI."""

    name: str

    @abc.abstractmethod
    def start(
        self,
        prompt: str,
        *,
        env: dict | None = None,
        timeout: float | None = None,
    ) -> AgentTurn:
        """Start a new session with ``prompt``. Must populate
        ``session_id`` if the agent supports resume."""

    @abc.abstractmethod
    def resume(
        self,
        prompt: str,
        *,
        session_id: str,
        env: dict | None = None,
        timeout: float | None = None,
    ) -> AgentTurn:
        """Continue session ``session_id`` with ``prompt``. Drivers
        whose agent doesn't natively support resume must emulate it
        (e.g. by replaying prior messages)."""


def _hermes_factory(**kwargs) -> AgentDriver:
    from .hermes import HermesDriver

    return HermesDriver(**kwargs)


def _opencode_factory(**kwargs) -> AgentDriver:
    from .opencode import OpenCodeDriver

    return OpenCodeDriver(**kwargs)


def _goose_factory(**kwargs) -> AgentDriver:
    from .goose import GooseDriver

    return GooseDriver(**kwargs)


def _pi_factory(**kwargs) -> AgentDriver:
    from .pi import PiDriver

    return PiDriver(**kwargs)


# Single source of truth for which agents the orchestrator supports.
# Adding a new driver: write the module + factory, append one entry
# here. Both ``get_driver`` and the ``--agent`` Click choice in
# ``__main__`` consume this — they cannot drift apart.
DRIVER_REGISTRY: dict[str, Callable[..., AgentDriver]] = {
    "hermes": _hermes_factory,
    "opencode": _opencode_factory,
    "goose": _goose_factory,
    "pi": _pi_factory,
}


def known_drivers() -> tuple[str, ...]:
    """Names of registered driver adapters, in registration order.

    Used to populate ``agentcap run --agent`` choices and to enumerate
    what's available without importing each driver module eagerly.
    """
    return tuple(DRIVER_REGISTRY)


def get_driver(name: str, **kwargs) -> AgentDriver:
    """Lookup a driver by short name."""
    try:
        factory = DRIVER_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown driver: {name!r}; known: {', '.join(known_drivers())}"
        ) from None
    return factory(**kwargs)


__all__ = [
    "AgentDriver",
    "AgentTurn",
    "DRIVER_REGISTRY",
    "get_driver",
    "known_drivers",
]
