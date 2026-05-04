"""Tests for the follow-up strategies."""

from __future__ import annotations

import pytest

from agentcap.followups import get_followup
from agentcap.followups.continue_ import ContinueFollowUp
from agentcap.followups.synthesized import SynthesizedFollowUp
from agentcap.followups.templates import TemplatesFollowUp


def test_continue_followup_always_returns_continue():
    fu = ContinueFollowUp()
    for turn in (2, 3, 100):
        assert (
            fu.next(original_task="anything", last_response="resp", turn=turn)
            == "continue"
        )


def test_continue_followup_custom_text():
    fu = ContinueFollowUp(text="more")
    assert fu.next(original_task="t", last_response="r", turn=2) == "more"


def test_templates_followup_rotates_through_pool():
    fu = TemplatesFollowUp(pool=("a", "b", "c"))
    seen = [
        fu.next(original_task="t", last_response="r", turn=t)
        for t in (2, 3, 4, 5, 6)
    ]
    assert seen == ["a", "b", "c", "a", "b"]


def test_templates_followup_default_pool_is_nonempty():
    fu = TemplatesFollowUp()
    out = fu.next(original_task="t", last_response="r", turn=2)
    assert isinstance(out, str) and out


def test_templates_followup_rejects_empty_pool():
    with pytest.raises(ValueError):
        TemplatesFollowUp(pool=())


def test_synthesized_followup_calls_synth_with_prompt():
    captured: dict = {}

    def fake_call(*, upstream, model, prompt, timeout):
        captured["upstream"] = upstream
        captured["model"] = model
        captured["prompt"] = prompt
        captured["timeout"] = timeout
        return "  Show me the migration plan.  "

    fu = SynthesizedFollowUp(
        upstream="http://synth:9000",
        model="synth-model",
        call=fake_call,
        timeout=10,
    )
    out = fu.next(
        original_task="Plan the S3 backend.",
        last_response="Here's a draft plan.",
        turn=2,
    )
    assert out == "Show me the migration plan."
    assert captured["upstream"] == "http://synth:9000"
    assert captured["model"] == "synth-model"
    assert captured["timeout"] == 10
    # Prompt embeds task and response
    assert "Plan the S3 backend." in captured["prompt"]
    assert "Here's a draft plan." in captured["prompt"]


def test_synthesized_followup_falls_back_on_exception():
    def boom(**_):
        raise RuntimeError("synth down")

    fu = SynthesizedFollowUp(
        upstream="http://synth", model="m", call=boom, fallback="continue"
    )
    assert fu.next(original_task="t", last_response="r", turn=2) == "continue"


def test_synthesized_followup_falls_back_on_empty_response():
    fu = SynthesizedFollowUp(
        upstream="http://synth",
        model="m",
        call=lambda **_: "   ",
        fallback="keep going",
    )
    assert fu.next(original_task="t", last_response="r", turn=2) == "keep going"


def test_get_followup_dispatch():
    assert isinstance(get_followup("continue"), ContinueFollowUp)
    assert isinstance(get_followup("templates"), TemplatesFollowUp)
    # synthesized requires upstream/model kwargs
    fu = get_followup(
        "synthesized", upstream="http://x", model="m", call=lambda **_: "ok"
    )
    assert isinstance(fu, SynthesizedFollowUp)


def test_get_followup_unknown():
    with pytest.raises(ValueError):
        get_followup("not-a-strategy")
