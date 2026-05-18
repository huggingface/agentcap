"""Pure-Python tests for ``agentcap.provider`` — classifier + hostname
fallback + HF Router sub-provider refinement. The actual network probe
is tested implicitly via the live integration suite; here we feed
synthetic ``endpoints`` dicts to exercise the classification logic."""

from __future__ import annotations

from agentcap.provider import (
    _classify,
    _hostname_fallback,
    refine_for_sub_provider,
)


# ---------------------------------------------------------------------------
# hostname fallback
# ---------------------------------------------------------------------------


def test_hostname_fallback_known_providers():
    assert _hostname_fallback("https://router.huggingface.co/v1") == "hf-router"
    assert _hostname_fallback("https://api.openai.com/v1") == "openai"
    assert _hostname_fallback("https://api.together.xyz/v1") == "together"
    assert _hostname_fallback("https://api.fireworks.ai/v1") == "fireworks"


def test_hostname_fallback_loopback_and_private():
    assert _hostname_fallback("http://127.0.0.1:8000/v1") == "local"
    assert _hostname_fallback("http://localhost:8000/v1") == "local"
    assert _hostname_fallback("http://10.0.0.5:8000/v1") == "local"
    assert _hostname_fallback("http://192.168.1.42:8000/v1") == "local"


def test_hostname_fallback_unknown_public():
    # eTLD+1-style: api.mycompany.com → "mycompany"
    assert _hostname_fallback("https://api.mycompany.com/v1") == "mycompany"


# ---------------------------------------------------------------------------
# classifier
# ---------------------------------------------------------------------------


def test_classify_hf_router_via_colon_suffix():
    endpoints = {
        "models": {"body": {"data": [
            {"id": "meta-llama/Llama-3.3-70B-Instruct"},
            {"id": "meta-llama/Llama-3.3-70B-Instruct:fireworks-ai"},
        ]}},
    }
    assert _classify(endpoints, "https://router.huggingface.co/v1") == "hf-router"


def test_classify_llama_cpp_via_props():
    endpoints = {
        "props": {"body": {"chat_template": "...", "n_ctx": 65536}},
        "models": {"body": {"data": [{"id": "qwen-test"}]}},
    }
    assert _classify(endpoints, "http://127.0.0.1:8000/v1") == "local-llama-server"


def test_classify_tgi_via_info_model_id():
    endpoints = {
        "info": {"body": {"model_id": "meta-llama/Llama-3.3-70B-Instruct",
                          "version": "2.4.1"}},
    }
    assert _classify(endpoints, "http://10.0.0.5:8000/v1") == "tgi"


def test_classify_vllm_via_version():
    endpoints = {
        "version": {"body": {"version": "0.7.0"}},
        "models": {"body": {"data": [{"id": "served-model"}]}},
    }
    assert _classify(endpoints, "http://10.0.0.5:8000/v1") == "vllm"


def test_classify_openai_via_model_ids():
    endpoints = {
        "models": {"body": {"data": [
            {"id": "gpt-4o-mini"},
            {"id": "o1-preview"},
        ]}},
    }
    assert _classify(endpoints, "https://api.openai.com/v1") == "openai"


def test_classify_falls_back_to_hostname_when_probe_empty():
    assert _classify({}, "https://router.huggingface.co/v1") == "hf-router"
    assert _classify({}, "http://127.0.0.1:8000/v1") == "local"


# ---------------------------------------------------------------------------
# refine_for_sub_provider
# ---------------------------------------------------------------------------


def test_refine_pins_hf_router_sub_provider():
    assert refine_for_sub_provider(
        "hf-router", "meta-llama/Llama-3.3-70B-Instruct:fireworks-ai"
    ) == "hf-router/fireworks-ai"


def test_refine_noop_without_colon_or_non_hf_router():
    assert refine_for_sub_provider("hf-router", "meta-llama/Llama-3.3-70B") == "hf-router"
    assert refine_for_sub_provider("local", "anything:fireworks-ai") == "local"


