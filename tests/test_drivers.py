"""Pure-Python tests for ``agentcap.drivers``.

These cover the *parser*, *config-builder*, and *overlay-scaffolding*
helpers — none of which shell out to a real agent. Live integration
tests for each driver (which actually invoke the agent CLI against a
running model server) live in ``test_drivers_live.py`` and are gated
on the agent binary being available in the sandbox image and on
``podman`` being on the host PATH.
"""

from __future__ import annotations


import pytest
import yaml

from agentcap.drivers import get_driver
from agentcap.drivers.goose import GooseDriver
from agentcap.drivers.hermes import (
    HermesDriver,
    _rewrite_config,
    parse_response_text as hermes_parse,
    parse_session_id,
)
from agentcap.drivers.opencode import (
    OpenCodeDriver,
    build_opencode_config,
    parse_response_text as opencode_parse,
    parse_session_id as opencode_parse_session,
)
from agentcap.drivers.pi import PiDriver, build_models_json


# ---------------------------------------------------------------------------
# Hermes parsers
# ---------------------------------------------------------------------------


def test_parse_session_id_finds_id():
    s = "blah\nsession_id: abc123_xyz\nmore\n"
    assert parse_session_id(s) == "abc123_xyz"


def test_parse_session_id_missing_returns_none():
    assert parse_session_id("nothing here") is None


def test_hermes_parse_response_initial_run():
    out = "Working on it...\nHere is the answer.\n"
    assert hermes_parse(out) == "Working on it...\nHere is the answer."


def test_hermes_parse_response_after_resumed_marker():
    out = (
        "↻ Resumed abc123\n"
        "old content\n"
        "↻ Resumed abc123\n"
        "the actual final answer\n"
        "across two lines\n"
    )
    assert hermes_parse(out) == "the actual final answer\nacross two lines"


def test_hermes_parse_response_strips_session_id_lines():
    out = "session_id: aa_bb\nactual response\n"
    assert hermes_parse(out) == "actual response"


# ---------------------------------------------------------------------------
# OpenCode parsers + config builder
# ---------------------------------------------------------------------------


def test_opencode_parse_concatenates_text_events():
    stdout = (
        '{"type":"step_start"}\n'
        '{"type":"text","text":"hello "}\n'
        '{"type":"text","text":"world"}\n'
        '{"type":"step_finish"}\n'
    )
    assert opencode_parse(stdout) == "hello world"


def test_opencode_parse_skips_malformed_lines():
    stdout = (
        "not json at all\n"
        '{"type":"text","text":"good"}\n'
        "\n"
    )
    assert opencode_parse(stdout) == "good"


def test_build_opencode_config_shape():
    cfg = build_opencode_config(
        provider_name="local",
        base_url="http://127.0.0.1:8001/v1",
        model_id="qwen-test",
    )
    prov = cfg["provider"]["local"]
    assert prov["options"]["baseURL"] == "http://127.0.0.1:8001/v1"
    assert "qwen-test" in prov["models"]
    assert cfg["model"] == "local/qwen-test"


# ---------------------------------------------------------------------------
# pi config builder
# ---------------------------------------------------------------------------


def test_pi_build_models_json_shape():
    payload = build_models_json(
        provider_name="local",
        base_url="http://127.0.0.1:8001/v1",
        model_id="qwen-test",
    )
    prov = payload["providers"]["local"]
    assert prov["baseUrl"] == "http://127.0.0.1:8001/v1"
    assert prov["api"] == "openai-completions"
    # llama.cpp's OpenAI shim doesn't accept the developer role pi
    # uses for reasoning-capable models — the config must downgrade.
    assert prov["compat"]["supportsDeveloperRole"] is False
    assert prov["compat"]["supportsReasoningEffort"] is False
    assert prov["models"][0]["id"] == "qwen-test"


# ---------------------------------------------------------------------------
# Driver registry + non-resumable driver behaviour
# ---------------------------------------------------------------------------


def test_get_driver_known_names(fake_sandbox):
    assert isinstance(get_driver("hermes", sandbox=fake_sandbox), HermesDriver)
    assert isinstance(get_driver("opencode", sandbox=fake_sandbox), OpenCodeDriver)
    assert isinstance(get_driver("goose", sandbox=fake_sandbox), GooseDriver)
    assert isinstance(get_driver("pi", sandbox=fake_sandbox), PiDriver)


def test_get_driver_unknown_name(fake_sandbox):
    with pytest.raises(ValueError):
        get_driver("not-a-real-driver", sandbox=fake_sandbox)


def test_opencode_parse_session_id_finds_top_level():
    stdout = (
        '{"type":"step_start","sessionID":"ses_abc123"}\n'
        '{"type":"text","text":"hi"}\n'
    )
    assert opencode_parse_session(stdout) == "ses_abc123"


def test_opencode_parse_session_id_finds_nested_under_part():
    stdout = (
        '{"type":"step_finish","timestamp":1,"part":{"sessionID":"ses_xyz"}}\n'
    )
    assert opencode_parse_session(stdout) == "ses_xyz"


def test_opencode_parse_session_id_missing_returns_none():
    assert opencode_parse_session('{"type":"text","text":"hi"}\n') is None


def test_hermes_driver_close_is_idempotent(fake_sandbox):
    drv = HermesDriver(sandbox=fake_sandbox)
    drv.close()
    drv.close()  # second call should not raise


# ---------------------------------------------------------------------------
# Hermes overlay HERMES_HOME (proxy_base_url support)
# ---------------------------------------------------------------------------


def test_rewrite_config_replaces_base_url_only():
    text = (
        "model:\n"
        "  provider: custom\n"
        "  base_url: http://localhost:8000/v1\n"
        "  key_env: OPENAI_API_KEY\n"
    )
    out = _rewrite_config(text, base_url="http://127.0.0.1:8001/v1")
    assert "base_url: http://127.0.0.1:8001/v1" in out
    assert "http://localhost:8000/v1" not in out
    # other keys preserved
    assert "key_env: OPENAI_API_KEY" in out
    assert "provider: custom" in out
    # no context_length added when override not requested
    assert "context_length" not in out


def test_rewrite_config_inserts_model_section_when_missing():
    out = _rewrite_config("", base_url="http://x:1/v1")
    assert "base_url: http://x:1/v1" in out


def test_rewrite_config_overrides_both_context_length_guards():
    """Hermes refuses startup if EITHER ``model.context_length`` or
    ``auxiliary.compression.context_length`` is below 64 K. The
    overlay must override both."""
    text = "model:\n  provider: custom\n  base_url: http://localhost:8000/v1\n"
    out = _rewrite_config(
        text,
        base_url="http://127.0.0.1:8001/v1",
        context_length_override=65536,
    )
    cfg = yaml.safe_load(out)
    assert cfg["model"]["context_length"] == 65536
    assert cfg["auxiliary"]["compression"]["context_length"] == 65536
    assert cfg["model"]["base_url"] == "http://127.0.0.1:8001/v1"


def test_rewrite_config_preserves_existing_auxiliary_keys():
    text = (
        "model:\n"
        "  provider: custom\n"
        "  base_url: http://localhost:8000/v1\n"
        "auxiliary:\n"
        "  compression:\n"
        "    model: my-compressor\n"
        "  other_key: keep_me\n"
    )
    out = _rewrite_config(
        text,
        base_url="http://x/v1",
        context_length_override=65536,
    )
    cfg = yaml.safe_load(out)
    assert cfg["auxiliary"]["compression"]["model"] == "my-compressor"
    assert cfg["auxiliary"]["compression"]["context_length"] == 65536
    assert cfg["auxiliary"]["other_key"] == "keep_me"


# NOTE: the host-side `build_overlay_hermes_home` function was
# removed when HermesDriver moved its overlay logic *inside* the
# sandbox. Behaviour previously verified by 5 unit tests against
# fake user-homes is now covered by the live driver test
# (tests/test_drivers_live.py::test_hermes_live) which exercises
# the full path against a real podman container. The pure-Python
# parts that survived as standalone helpers (`_rewrite_config`)
# keep their own tests above.
