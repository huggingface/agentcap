"""Detect the inference backend behind ``--upstream``.

Most OpenAI-compat servers expose richer self-description than just
``/v1/models``. :func:`probe` issues a small set of parallel GETs to
well-known introspection endpoints, unions whatever returns, and
classifies via distinctive markers:

- llama.cpp:  ``GET /props``    — chat template, n_ctx, model alias
- TGI:        ``GET /info``     — model_id, model_sha, dtype, version
- vLLM:       ``GET /version``  — engine version
- HF Router:  ``GET /v1/models`` returns ids with ``:<sub-provider>``
- OpenAI:     ``GET /v1/models`` returns ``gpt-*`` / ``o*-*`` ids
- generic:    ``GET /v1/models`` only

Result lands verbatim in ``<trace_dir>/_meta.json`` so consumers and
the export pipeline see the same fingerprint the orchestrator saw at
run time. :func:`flatten_for_parquet` picks the small set of fields
worth promoting to parquet columns.

Never raises: an unreachable upstream returns ``{"provider": …
hostname fallback …, "endpoints": {}}`` so ``agentcap run`` doesn't
abort on probe failure.
"""

from __future__ import annotations

import concurrent.futures
import ipaddress
import time
from typing import Any
from urllib.parse import urlparse

import httpx


# Hostnames that map to a known provider when probing fails or returns
# nothing classifiable. Reverse proxies / custom domains won't match;
# probing covers those.
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
    """Strip a trailing ``/v1`` to get the server root that introspection
    endpoints (``/props``, ``/info``, …) live under."""
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
    """One probe. Returns ``{"body": json_or_None, "text": ..., "headers": ...}``
    on a 2xx, ``None`` otherwise (or on any transport error)."""
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
            # Prometheus / plain-text endpoints (llama.cpp + vLLM metrics)
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
    """Probe an OpenAI-compat upstream and return a self-describing
    metadata dict — always returns, never raises."""
    root = _base_root(upstream_url)
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    targets = {
        "props":   f"{root}/props",          # llama.cpp
        "info":    f"{root}/info",           # TGI
        "version": f"{root}/version",        # vLLM
        "models":  f"{root}/v1/models",      # universal OpenAI-compat
        "metrics": f"{root}/metrics",        # llama.cpp + vLLM (text)
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

    # HF Router: ``meta-llama/Llama-3.3-70B-Instruct:fireworks-ai`` —
    # the ``:provider`` suffix is unique to it.
    if any(":" in i for i in model_ids):
        return "hf-router"

    # llama.cpp's /props endpoint exposes a chat template and is unique
    # to it among OpenAI-compat backends.
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
    """HF Router lets you pin a sub-provider in the model id itself
    (``meta-llama/Llama-3.3-70B-Instruct:fireworks-ai``). Surface
    that pin as ``hf-router/<sub>`` so the parquet's provider column
    distinguishes captures routed through different sub-providers."""
    if provider == "hf-router" and model and ":" in model:
        return f"hf-router/{model.split(':', 1)[1]}"
    return provider


def flatten_for_parquet(meta: dict) -> dict:
    """Pick the small set of probe fields worth promoting to columns.

    Explicit top-level keys (``server_version``, ``served_model_id``)
    win over endpoint-derived values — this lets retroactive scripts
    that rebuild ``_meta.json`` for historical captures inject a known
    version without having to fake the full endpoint structure.

    The raw ``endpoints`` dict stays in ``_meta.json`` for forensics."""
    endpoints = meta.get("endpoints") or {}
    return {
        "provider": meta.get("provider", "unknown"),
        "upstream_url": meta.get("upstream_url", ""),
        "server_version": (
            meta.get("server_version")
            or _extract_server_version(endpoints)
        ),
        "served_model_id": (
            meta.get("served_model_id")
            or _extract_served_model(endpoints)
        ),
    }


def _extract_server_version(endpoints: dict) -> str:
    info = (endpoints.get("info") or {}).get("body") or {}
    if isinstance(info, dict) and info.get("version"):
        return f"tgi {info['version']}"
    version = (endpoints.get("version") or {}).get("body") or {}
    if isinstance(version, dict) and version.get("version"):
        return f"vllm {version['version']}"
    metrics = (endpoints.get("metrics") or {}).get("text") or ""
    # llama.cpp metrics include a build-version comment near the top.
    for line in metrics.splitlines()[:50]:
        s = line.strip()
        if s.startswith("# HELP") and "llamacpp" in s.lower():
            return "llama.cpp"
    return ""


def _extract_served_model(endpoints: dict) -> str:
    info = (endpoints.get("info") or {}).get("body") or {}
    if isinstance(info, dict) and info.get("model_id"):
        return info["model_id"]
    props = (endpoints.get("props") or {}).get("body") or {}
    if isinstance(props, dict):
        gen = props.get("default_generation_settings") or {}
        if isinstance(gen, dict) and gen.get("model"):
            return gen["model"]
    models = (endpoints.get("models") or {}).get("body") or {}
    data = models.get("data") or []
    if data and isinstance(data[0], dict) and data[0].get("id"):
        return data[0]["id"]
    return ""
