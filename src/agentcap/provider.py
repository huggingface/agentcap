"""Identify the inference backend behind an upstream URL.

Hostname classification (:func:`_hostname_fallback`) +
HF Router sub-provider pin (:func:`refine_for_sub_provider`).
:func:`probe` is the richer (network) variant — issues parallel
GETs to well-known introspection endpoints, never raises.
"""

from __future__ import annotations

import concurrent.futures
import ipaddress
import time
from typing import Any
from urllib.parse import urlparse

import httpx


# Reverse proxies / custom domains won't match; the probe path catches those.
_HOSTNAME_TO_PROVIDER: dict[str, str] = {
    "router.huggingface.co": "hf-router",
    "api.openai.com": "openai",
    "api.together.xyz": "together",
    "api.anthropic.com": "anthropic",
    "api.cerebras.ai": "cerebras",
    "api.fireworks.ai": "fireworks",
    "api.groq.com": "groq",
}


def _base_root(upstream_url: str) -> str:
    # Introspection endpoints (/props, /info, ...) live under the
    # server root, not /v1.
    base = upstream_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    return base


def _hostname_fallback(upstream_url: str) -> str:
    host = (urlparse(upstream_url).hostname or "").lower()
    if not host:
        return "unknown"
    if host in _HOSTNAME_TO_PROVIDER:
        return _HOSTNAME_TO_PROVIDER[host]
    if host in ("localhost", "::1"):
        return "local"
    try:
        ip = ipaddress.ip_address(host)
        return "local" if (ip.is_loopback or ip.is_private) else host
    except ValueError:
        pass
    parts = host.split(".")
    return parts[-2] if len(parts) >= 2 else host


def _try_get(url: str, headers: dict, timeout: float) -> dict | None:
    try:
        r = httpx.get(url, headers=headers, timeout=timeout)
    except (httpx.HTTPError, OSError):
        return None
    if r.status_code != 200:
        return None
    ct = r.headers.get("content-type", "")
    out: dict[str, Any] = {"headers": {k.lower(): v for k, v in r.headers.items()}}
    try:
        if "json" in ct:
            out["body"] = r.json()
        else:
            out["text"] = r.text[:4096]
    except Exception:
        return None
    return out


def probe(
    upstream_url: str,
    *,
    api_key: str | None = None,
    timeout: float = 3.0,
) -> dict:
    """Probe an OpenAI-compat upstream. Never raises."""
    root = _base_root(upstream_url)
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    targets = {
        "props":   f"{root}/props",          # llama.cpp
        "info":    f"{root}/info",           # TGI
        "version": f"{root}/version",        # vLLM
        "models":  f"{root}/v1/models",
        "metrics": f"{root}/metrics",
    }
    endpoints: dict[str, dict] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(targets)) as pool:
        futures = {
            name: pool.submit(_try_get, url, headers, timeout)
            for name, url in targets.items()
        }
        for name, fut in futures.items():
            try:
                res = fut.result(timeout=timeout + 1.0)
            except concurrent.futures.TimeoutError:
                res = None
            if res is not None:
                endpoints[name] = res

    return {
        "upstream_url": upstream_url,
        "provider": _classify(endpoints, upstream_url),
        "probed_at": int(time.time()),
        "endpoints": endpoints,
    }


def _classify(endpoints: dict, upstream_url: str) -> str:
    models_body = (endpoints.get("models") or {}).get("body") or {}
    model_ids = [m.get("id", "") for m in (models_body.get("data") or [])]

    # HF Router model ids carry a ``:<sub-provider>`` suffix.
    if any(":" in i for i in model_ids):
        return "hf-router"
    if endpoints.get("props") is not None:
        return "local-llama-server"

    info_body = (endpoints.get("info") or {}).get("body") or {}
    if isinstance(info_body, dict) and info_body.get("model_id"):
        return "tgi"

    version_body = (endpoints.get("version") or {}).get("body") or {}
    if isinstance(version_body, dict) and version_body.get("version"):
        return "vllm"

    if any(i.startswith(("gpt-", "o1-", "o3-", "o4-")) for i in model_ids):
        return "openai"

    return _hostname_fallback(upstream_url)


def refine_for_sub_provider(provider: str, model: str | None) -> str:
    """Surface HF Router's ``meta-llama/...:fireworks-ai`` pin as
    ``hf-router/fireworks-ai`` in the provider slug."""
    if provider == "hf-router" and model and ":" in model:
        return f"hf-router/{model.split(':', 1)[1]}"
    return provider
