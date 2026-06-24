"""Tests for the orchestrator."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from agentcap.drivers import AgentDriver, AgentTurn
from agentcap.followups.continue_ import ContinueFollowUp
from agentcap.followups.templates import TemplatesFollowUp
from agentcap.orchestrator import Orchestrator, read_tasks_txt


# ---------------------------------------------------------------------------
# Fake driver
# ---------------------------------------------------------------------------


class FakeDriver(AgentDriver):
    """Records every call; returns scripted AgentTurns."""

    name = "fake"

    def __init__(
        self,
        *,
        start_turn: AgentTurn | None = None,
        resume_turn: AgentTurn | None = None,
        resume_unsupported: bool = False,
    ) -> None:
        self.calls: list[tuple[str, str, str | None]] = []  # (op, prompt, sid)
        self._start_turn = start_turn or AgentTurn(
            session_id="ses_fake", response_text="initial response", returncode=0,
            stdout="hi", stderr="",
        )
        self._resume_turn = resume_turn or AgentTurn(
            session_id="ses_fake", response_text="continuation", returncode=0,
            stdout="ok", stderr="",
        )
        self._resume_unsupported = resume_unsupported

    def start(self, prompt, *, env=None, timeout=None):
        self.calls.append(("start", prompt, None))
        return self._start_turn

    def resume(self, prompt, *, session_id, env=None, timeout=None):
        self.calls.append(("resume", prompt, session_id))
        if self._resume_unsupported:
            raise NotImplementedError("fake doesn't resume")
        return self._resume_turn


# ---------------------------------------------------------------------------
# read_tasks_txt
# ---------------------------------------------------------------------------


def test_read_tasks_skips_comments_and_blanks(tmp_path: Path):
    p = tmp_path / "tasks.txt"
    p.write_text(
        "# header comment\n"
        "first task\n"
        "\n"
        "  # indented comment\n"
        "second task\n"
        "  third task with leading space\n"
    )
    tasks = read_tasks_txt(p)
    assert tasks == ["first task", "second task", "third task with leading space"]


def test_read_tasks_empty_file(tmp_path: Path):
    p = tmp_path / "tasks.txt"
    p.write_text("# only comments\n\n")
    assert read_tasks_txt(p) == []


# ---------------------------------------------------------------------------
# Orchestrator.run_task
# ---------------------------------------------------------------------------


def test_run_task_single_turn_no_followup_call():
    drv = FakeDriver()
    fu = ContinueFollowUp()
    orch = Orchestrator(drv, fu)

    result = orch.run_task("Plan the S3 backend", task_id="t01", turns=1)
    assert len(result.turns) == 1
    assert result.turns[0].turn == 1
    assert result.turns[0].prompt == "Plan the S3 backend"
    assert result.session_id == "ses_fake"
    # Driver was called once for start, never for resume
    assert [c[0] for c in drv.calls] == ["start"]


def test_run_task_multi_turn_uses_continue_followup():
    drv = FakeDriver()
    orch = Orchestrator(drv, ContinueFollowUp())
    result = orch.run_task("task", task_id="t01", turns=4)
    assert len(result.turns) == 4
    assert [c[0] for c in drv.calls] == ["start", "resume", "resume", "resume"]
    # All resume prompts are "continue"
    for op, prompt, sid in drv.calls[1:]:
        assert prompt == "continue"
        assert sid == "ses_fake"


def test_run_task_multi_turn_uses_templates_pool():
    drv = FakeDriver()
    pool = ("first", "second", "third")
    orch = Orchestrator(drv, TemplatesFollowUp(pool=pool))
    orch.run_task("task", task_id="t01", turns=4)
    # Skip the start call; resume prompts cycle through pool
    resume_prompts = [p for op, p, _ in drv.calls if op == "resume"]
    assert resume_prompts == list(pool)


def test_run_task_aborts_when_initial_returncode_nonzero():
    drv = FakeDriver(
        start_turn=AgentTurn(
            session_id=None, response_text="", returncode=1, stdout="", stderr="boom"
        )
    )
    orch = Orchestrator(drv, ContinueFollowUp())
    result = orch.run_task("task", task_id="t01", turns=3)
    assert len(result.turns) == 1
    assert result.completed_turns == 0
    # No resume calls were made
    assert all(c[0] == "start" for c in drv.calls)


def test_run_task_aborts_when_no_session_id_for_multi_turn():
    drv = FakeDriver(
        start_turn=AgentTurn(
            session_id=None, response_text="hi", returncode=0, stdout="", stderr=""
        )
    )
    orch = Orchestrator(drv, ContinueFollowUp())
    result = orch.run_task("task", task_id="t01", turns=3)
    assert len(result.turns) == 1
    # Only the start call; resume never happens because session_id is None
    assert all(c[0] == "start" for c in drv.calls)


def test_run_task_breaks_loop_on_resume_failure():
    drv = FakeDriver(
        resume_turn=AgentTurn(
            session_id="ses_fake", response_text="", returncode=124,
            stdout="", stderr="timeout",
        )
    )
    orch = Orchestrator(drv, ContinueFollowUp())
    result = orch.run_task("task", task_id="t01", turns=4)
    # One success + one failure, then the loop breaks
    assert len(result.turns) == 2
    assert result.turns[1].returncode == 124


def test_run_task_handles_resume_not_implemented():
    drv = FakeDriver(resume_unsupported=True)
    orch = Orchestrator(drv, ContinueFollowUp())
    result = orch.run_task("task", task_id="t01", turns=3)
    # First turn succeeds; resume raises NotImplementedError; orchestrator stops
    assert len(result.turns) == 1


def test_run_task_rejects_zero_turns():
    drv = FakeDriver()
    orch = Orchestrator(drv, ContinueFollowUp())
    with pytest.raises(ValueError):
        orch.run_task("t", task_id="x", turns=0)


def test_run_task_writes_session_logs_when_sessions_dir_set(tmp_path: Path):
    drv = FakeDriver(
        start_turn=AgentTurn(
            session_id="s1", response_text="r", returncode=0,
            stdout="STDOUT-init", stderr="STDERR-init",
        ),
        resume_turn=AgentTurn(
            session_id="s1", response_text="r2", returncode=0,
            stdout="STDOUT-cont", stderr="STDERR-cont",
        ),
    )
    sessions = tmp_path / "sessions"
    orch = Orchestrator(drv, ContinueFollowUp(), sessions_dir=sessions)
    orch.run_task("t", task_id="task_01", turns=2)

    assert (sessions / "task_01_turn_01.out").read_text() == "STDOUT-init"
    assert (sessions / "task_01_turn_01.err").read_text() == "STDERR-init"
    assert (sessions / "task_01_turn_02.out").read_text() == "STDOUT-cont"
    assert (sessions / "task_01_turn_02.err").read_text() == "STDERR-cont"


# ---------------------------------------------------------------------------
# Orchestrator.run_corpus
# ---------------------------------------------------------------------------


def test_run_corpus_iterates_tasks_with_default_id_format():
    drv = FakeDriver()
    orch = Orchestrator(drv, ContinueFollowUp())
    results = orch.run_corpus(
        ["task A", "task B", "task C"], turns_per_task=1
    )
    assert [r.task_id for r in results] == ["task_01", "task_02", "task_03"]
    assert [r.prompt for r in results] == ["task A", "task B", "task C"]


def test_run_corpus_records_events():
    drv = FakeDriver()
    events: list[tuple[str, dict]] = []

    def listener(event: str, **kw):
        events.append((event, kw))

    orch = Orchestrator(drv, ContinueFollowUp(), on_event=listener)
    orch.run_corpus(["task A"], turns_per_task=2)
    event_names = [e for e, _ in events]
    assert event_names[0] == "task_start"
    assert event_names.count("turn_done") == 2


def _timeout_after_n(n: int):
    """Return a driver whose start/resume raises TimeoutExpired on the
    n-th call (1-indexed), succeeds otherwise."""
    import subprocess

    class TimeoutDriver(FakeDriver):
        def __init__(self):
            super().__init__()
            self._n = 0

        def start(self, prompt, *, env=None, timeout=None):
            self._n += 1
            if self._n == n:
                raise subprocess.TimeoutExpired(["fake"], timeout or 1)
            return super().start(prompt, env=env, timeout=timeout)

        def resume(self, prompt, *, session_id, env=None, timeout=None):
            self._n += 1
            if self._n == n:
                raise subprocess.TimeoutExpired(["fake"], timeout or 1)
            return super().resume(prompt, session_id=session_id, env=env, timeout=timeout)

    return TimeoutDriver()


def test_run_task_aborts_on_initial_turn_timeout():
    """A driver timeout on turn 1 must not propagate; the task is
    aborted with a recorded event and ``run_corpus`` keeps going."""
    drv = _timeout_after_n(1)
    events: list[tuple[str, dict]] = []
    orch = Orchestrator(
        drv, ContinueFollowUp(), on_event=lambda **kw: events.append((kw.pop("event"), kw))
    )
    result = orch.run_task("anything", task_id="t01", turns=2)
    assert result.turns == []
    aborted = [e for e in events if e[0] == "task_aborted"]
    assert aborted and aborted[0][1]["reason"] == "initial-turn-timeout"


def test_run_corpus_keeps_going_when_one_task_times_out():
    """Critical: a timeout on task 1 must not kill tasks 2+."""
    # Total calls across the run: t1 start (timeout), t2 start (ok),
    # t3 start (ok). Trip the 1st call only.
    drv = _timeout_after_n(1)
    orch = Orchestrator(drv, ContinueFollowUp())
    results = orch.run_corpus(
        ["task A", "task B", "task C"], turns_per_task=1
    )
    assert len(results) == 3
    # task A failed before any turn could be recorded
    assert results[0].turns == []
    # tasks B and C completed turn 1
    assert len(results[1].turns) == 1
    assert len(results[2].turns) == 1


def test_run_task_aborts_on_followup_turn_timeout():
    drv = _timeout_after_n(2)  # 1st call ok (start), 2nd (resume) times out
    events: list[tuple[str, dict]] = []
    orch = Orchestrator(
        drv, ContinueFollowUp(), on_event=lambda **kw: events.append((kw.pop("event"), kw))
    )
    result = orch.run_task("anything", task_id="t01", turns=3)
    # Only turn 1 recorded.
    assert len(result.turns) == 1
    aborted = [e for e in events if e[0] == "task_aborted"]
    assert aborted and aborted[0][1]["reason"] == "follow-up-turn-timeout"

def test_read_tasks_yaml_list_of_strings(tmp_path: Path):
    pytest.importorskip("yaml")
    p = tmp_path / "tasks.yaml"
    p.write_text("- first task\n- second task\n")
    assert read_tasks_txt(p) == ["first task", "second task"]


def test_read_tasks_yaml_mapping_with_task_objects(tmp_path: Path):
    pytest.importorskip("yaml")
    p = tmp_path / "tasks.yml"
    p.write_text(
        "tasks:\n"
        "  - prompt: |\n"
        "      first line\n"
        "      second line\n"
        "  - second task\n"
    )
    assert read_tasks_txt(p) == ["first line\nsecond line", "second task"]


def test_read_tasks_yaml_validates_shape(tmp_path: Path):
    pytest.importorskip("yaml")
    p = tmp_path / "tasks.yaml"
    p.write_text("name: not-a-task-list\n")
    with pytest.raises(ValueError, match="top-level 'tasks' key"):
        read_tasks_txt(p)


def test_read_tasks_yaml_requires_pyyaml(monkeypatch, tmp_path: Path):
    p = tmp_path / "tasks.yaml"
    p.write_text("- first task\n")
    monkeypatch.setitem(sys.modules, "yaml", None)
    with pytest.raises(RuntimeError, match="YAML task files require PyYAML"):
        read_tasks_txt(p)