"""Drive an agent CLI through a corpus of prompts.

The orchestrator pairs an :class:`AgentDriver` with a :class:`FollowUp`
strategy and steps each task through ``turns_per_task`` turns. The
proxy that captures the actual chat-completion bytes is configured
separately (started before the orchestrator runs and pointed at via
the agent's own config); this module is intentionally proxy-agnostic.

Per-turn driver stdout/stderr is written under
``<sessions_dir>/task_<NN>_turn_<K>.{out,err}`` for debugging. The
orchestrator's primary output is the list of :class:`TaskResult`
objects returned by :meth:`Orchestrator.run_corpus`.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

from .drivers import AgentDriver, AgentTurn
from .followups import FollowUp


@dataclass
class TaskTurnResult:
    turn: int                     # 1-indexed
    prompt: str
    session_id: str | None
    returncode: int
    response_text: str
    duration_s: float


@dataclass
class TaskResult:
    task_id: str
    prompt: str
    turns: list[TaskTurnResult] = field(default_factory=list)

    @property
    def session_id(self) -> str | None:
        if self.turns:
            return self.turns[0].session_id
        return None

    @property
    def completed_turns(self) -> int:
        return sum(1 for t in self.turns if t.returncode == 0)


def read_tasks_txt(path: Path | str) -> list[str]:
    """Read a plain-text tasks file (one prompt per line, ``#`` comments
    and blank lines ignored)."""
    text = Path(path).read_text()
    out: list[str] = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.append(s)
    return out


class Orchestrator:
    """Run a corpus through an agent driver with a follow-up strategy."""

    def __init__(
        self,
        driver: AgentDriver,
        followup: FollowUp,
        *,
        sessions_dir: Path | str | None = None,
        set_capture_context: Callable[..., None] | None = None,
        on_event: Callable[..., None] | None = None,
    ) -> None:
        self.driver = driver
        self.followup = followup
        self.sessions_dir = Path(sessions_dir) if sessions_dir else None
        self.set_capture_context = set_capture_context or (lambda **_: None)
        if self.sessions_dir is not None:
            self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.on_event = on_event or (lambda **_: None)

    def _log_turn(self, task_id: str, turn: int, agent_turn: AgentTurn) -> None:
        if self.sessions_dir is None:
            return
        base = self.sessions_dir / f"{task_id}_turn_{turn:02d}"
        base.with_suffix(".out").write_text(agent_turn.stdout)
        base.with_suffix(".err").write_text(agent_turn.stderr)

    def run_task(
        self,
        prompt: str,
        *,
        task_id: str,
        turns: int,
        timeout: float | None = None,
    ) -> TaskResult:
        if turns < 1:
            raise ValueError("turns must be >= 1")

        result = TaskResult(task_id=task_id, prompt=prompt)

        # Turn 1: open session
        self.on_event(event="task_start", task_id=task_id, prompt=prompt, turns=turns)
        self.set_capture_context(task_id=task_id, turn=1)
        t0 = time.time()
        try:
            first = self.driver.start(prompt, timeout=timeout)
        except subprocess.TimeoutExpired:
            dur = time.time() - t0
            self.on_event(
                event="task_aborted",
                task_id=task_id,
                reason="initial-turn-timeout",
                duration_s=dur,
            )
            return result
        dur = time.time() - t0
        result.turns.append(
            TaskTurnResult(
                turn=1,
                prompt=prompt,
                session_id=first.session_id,
                returncode=first.returncode,
                response_text=first.response_text,
                duration_s=dur,
            )
        )
        self._log_turn(task_id, 1, first)
        self.on_event(
            event="turn_done",
            task_id=task_id,
            turn=1,
            session_id=first.session_id,
            returncode=first.returncode,
            duration_s=dur,
        )

        if first.returncode != 0:
            self.on_event(event="task_aborted", task_id=task_id, reason="initial-turn-failed")
            return result
        if first.session_id is None and turns > 1:
            self.on_event(event="task_aborted", task_id=task_id, reason="no-session-id")
            return result

        # Follow-up turns
        last_response = first.response_text
        sid = first.session_id
        for turn in range(2, turns + 1):
            next_prompt = self.followup.next(
                original_task=prompt, last_response=last_response, turn=turn
            )
            self.set_capture_context(task_id=task_id, turn=turn)
            t0 = time.time()
            try:
                fu = self.driver.resume(next_prompt, session_id=sid, timeout=timeout)
            except NotImplementedError:
                self.on_event(
                    event="task_aborted",
                    task_id=task_id,
                    reason="resume-not-supported",
                )
                break
            except subprocess.TimeoutExpired:
                dur = time.time() - t0
                self.on_event(
                    event="task_aborted",
                    task_id=task_id,
                    reason="follow-up-turn-timeout",
                    turn=turn,
                    duration_s=dur,
                )
                break
            dur = time.time() - t0
            result.turns.append(
                TaskTurnResult(
                    turn=turn,
                    prompt=next_prompt,
                    session_id=sid,
                    returncode=fu.returncode,
                    response_text=fu.response_text,
                    duration_s=dur,
                )
            )
            self._log_turn(task_id, turn, fu)
            self.on_event(
                event="turn_done",
                task_id=task_id,
                turn=turn,
                session_id=sid,
                returncode=fu.returncode,
                duration_s=dur,
            )
            if fu.returncode != 0:
                break
            last_response = fu.response_text

        return result

    def run_corpus(
        self,
        tasks: Sequence[str] | Iterable[str],
        *,
        turns_per_task: int,
        timeout: float | None = None,
        task_id_format: str = "task_{i:02d}",
    ) -> list[TaskResult]:
        results: list[TaskResult] = []
        for i, prompt in enumerate(tasks, start=1):
            tid = task_id_format.format(i=i)
            results.append(
                self.run_task(
                    prompt, task_id=tid, turns=turns_per_task, timeout=timeout
                )
            )
        return results
