"""Pure-Python tests for ``agentcap.drivers``.

These cover the *parser*, *config-builder*, and *overlay-scaffolding*
helpers — none of which shell out to a real agent. Live integration
tests for each driver (which actually invoke the agent CLI against a
running model server) live in ``test_drivers_live.py`` and are gated
on the agent binary being on PATH plus an ``AGENTCAP_TEST_LLM_URL``
endpoint.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from agentcap.drivers import get_driver
from agentcap.drivers.goose import GooseDriver
from agentcap.drivers.hermes import (
    HermesDriver,
    _rewrite_config,
    build_overlay_hermes_home,
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


def test_get_driver_known_names():
    assert isinstance(get_driver("hermes"), HermesDriver)
    assert isinstance(get_driver("opencode"), OpenCodeDriver)
    assert isinstance(get_driver("goose"), GooseDriver)
    assert isinstance(get_driver("pi"), PiDriver)


def test_get_driver_unknown_name():
    with pytest.raises(ValueError):
        get_driver("not-a-real-driver")


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


def test_hermes_driver_close_is_idempotent():
    drv = HermesDriver()
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


def _make_user_hermes_home(root: Path) -> Path:
    home = root / "user_hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "model:\n  provider: custom\n  base_url: http://localhost:8000/v1\n"
    )
    # Identity / read-only content (overlay symlinks these through)
    (home / "skills").mkdir()
    (home / "skills" / "demo.md").write_text("a skill")
    (home / "SOUL.md").write_text("# soul")
    # Writable per-run state (overlay isolates these)
    (home / "memories").mkdir()
    (home / "memories" / "MEMORY.md").write_text(
        "user's persistent memory — must not be touched\n"
    )
    (home / "sessions").mkdir()
    (home / "sessions" / "old.json").write_text("{}")
    (home / "state.db").write_text("user state bytes")
    return home


def test_build_overlay_symlinks_identity_content(tmp_path: Path):
    user_home = _make_user_hermes_home(tmp_path)
    overlay = build_overlay_hermes_home(
        "http://127.0.0.1:8001/v1",
        user_hermes_home=user_home,
        overlay_root=tmp_path / "overlay",
    )
    cfg = overlay / "config.yaml"
    assert cfg.is_file() and not cfg.is_symlink()
    assert "http://127.0.0.1:8001/v1" in cfg.read_text()
    assert (overlay / "skills").is_symlink()
    assert (overlay / "skills").resolve() == (user_home / "skills").resolve()
    assert (overlay / "skills" / "demo.md").read_text() == "a skill"
    assert (overlay / "SOUL.md").is_symlink()


def test_build_overlay_isolates_writable_state(tmp_path: Path):
    """Memory / session / state.db / sandbox are NOT symlinked through —
    they're per-run-fresh so a capture run never reads or mutates the
    user's persistent ~/.hermes state."""
    user_home = _make_user_hermes_home(tmp_path)
    overlay = build_overlay_hermes_home(
        "http://127.0.0.1:8001/v1",
        user_hermes_home=user_home,
        overlay_root=tmp_path / "overlay",
    )
    mem = overlay / "memories"
    assert mem.is_dir() and not mem.is_symlink()
    assert list(mem.iterdir()) == []
    sess = overlay / "sessions"
    assert sess.is_dir() and not sess.is_symlink()
    assert list(sess.iterdir()) == []
    assert not (overlay / "state.db").is_symlink()
    if (overlay / "state.db").exists():
        assert (overlay / "state.db").read_text() != "user state bytes"


def test_build_overlay_writes_to_overlay_dont_leak_to_user_home(tmp_path: Path):
    """Concrete check: writing into overlay/memories/ must not surface
    in user_home/memories/. Regression test for the memory-bleed bug."""
    user_home = _make_user_hermes_home(tmp_path)
    overlay = build_overlay_hermes_home(
        "http://127.0.0.1:8001/v1",
        user_hermes_home=user_home,
        overlay_root=tmp_path / "overlay",
    )
    (overlay / "memories" / "MEMORY.md").write_text(
        "agentcap-internal note — must not escape\n"
    )
    user_mem = (user_home / "memories" / "MEMORY.md").read_text()
    assert "agentcap-internal" not in user_mem
    assert user_mem == "user's persistent memory — must not be touched\n"


def test_build_overlay_missing_user_home_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        build_overlay_hermes_home(
            "http://x", user_hermes_home=tmp_path / "nope"
        )


def test_build_overlay_missing_user_config_raises(tmp_path: Path):
    home = tmp_path / "home"
    home.mkdir()
    with pytest.raises(FileNotFoundError):
        build_overlay_hermes_home(
            "http://x", user_hermes_home=home, overlay_root=tmp_path / "ov"
        )
