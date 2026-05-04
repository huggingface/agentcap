"""Shared fixtures.

Live driver tests need an agent binary plus a reachable OpenAI-compat
``/v1`` endpoint. Both are env-gated; missing prereqs mean the test
skips. See README ``Running tests`` for full setup.

Endpoint resolution, in order:

  1. ``AGENTCAP_TEST_LLM_URL`` set         -> use it as-is.
  2. ``AGENTCAP_TEST_GGUF`` + ``llama-server`` (or
     ``AGENTCAP_TEST_LLAMA_BIN``)          -> spin a session-scoped
                                              llama-server on a free
                                              port, tear down on exit.
  3. otherwise                             -> skip.

Env vars:

  ``AGENTCAP_TEST_LLM_URL``     OpenAI-compat /v1 base URL
  ``AGENTCAP_TEST_GGUF``        GGUF path for auto-bootstrap
  ``AGENTCAP_TEST_LLAMA_BIN``   path to ``llama-server``
  ``AGENTCAP_TEST_NGL``         ``--n-gpu-layers`` (default 999; 0=CPU)
  ``AGENTCAP_TEST_CTX_SIZE``    llama-server ctx-size (default 8192)
  ``AGENTCAP_TEST_MODEL``       model alias agents send
                                (default qwen3.6-35b-a3b)
  ``AGENTCAP_TEST_<AGENT>_BIN`` per-agent binary override
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import time
from pathlib import Path
from urllib.request import urlopen

import pytest


def _resolve_binary(name: str) -> str | None:
    env = os.environ.get(f"AGENTCAP_TEST_{name.upper()}_BIN")
    if env:
        return env if Path(env).is_file() else None
    return shutil.which(name)


def _require_binary(name: str) -> str:
    path = _resolve_binary(name)
    if path is None:
        pytest.skip(
            f"{name!r} binary not found "
            f"(set AGENTCAP_TEST_{name.upper()}_BIN or install on PATH)"
        )
    return path


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_ready(url: str, timeout: float = 180.0) -> None:
    """Poll a ``/v1/models`` endpoint until it responds 200 or we
    blow ``timeout`` seconds. Tiny GGUFs load in seconds; the budget
    is generous so we don't flake on a cold weight load."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError(f"llama-server never became ready at {url}")


@pytest.fixture(scope="session")
def live_proxy_base_url():
    """OpenAI-compat ``/v1`` URL the agent will hit.

    If ``AGENTCAP_TEST_LLM_URL`` is set, return it (caller is
    responsible for the server). Otherwise, if ``AGENTCAP_TEST_GGUF``
    is set and ``llama-server`` is on PATH, spin one up for the
    duration of the pytest session. Otherwise skip.
    """
    url = os.environ.get("AGENTCAP_TEST_LLM_URL")
    if url:
        yield url
        return

    gguf = os.environ.get("AGENTCAP_TEST_GGUF")
    llama = os.environ.get("AGENTCAP_TEST_LLAMA_BIN") or shutil.which(
        "llama-server"
    )
    if not gguf or not llama:
        pytest.skip(
            "set AGENTCAP_TEST_LLM_URL (existing endpoint) OR "
            "AGENTCAP_TEST_GGUF=<path> with llama-server on PATH "
            "to enable live driver tests"
        )

    port = _free_port()
    ctx = os.environ.get("AGENTCAP_TEST_CTX_SIZE", "8192")
    ngl = os.environ.get("AGENTCAP_TEST_NGL", "999")
    argv = [
        llama,
        "--model",
        gguf,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--ctx-size",
        ctx,
        "--reasoning",
        "off",
        "--jinja",
        "--n-gpu-layers",
        ngl,
    ]
    log = open("/tmp/agentcap-pytest-llama.log", "w")
    proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}/v1"
    try:
        _wait_ready(base + "/models", timeout=180)
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()


@pytest.fixture(scope="session")
def live_model() -> str:
    return os.environ.get("AGENTCAP_TEST_MODEL", "qwen3.6-35b-a3b")


@pytest.fixture(scope="session")
def goose_bin() -> str:
    return _require_binary("goose")


@pytest.fixture(scope="session")
def pi_bin() -> str:
    return _require_binary("pi")


@pytest.fixture(scope="session")
def opencode_bin() -> str:
    return _require_binary("opencode")


@pytest.fixture(scope="session")
def hermes_bin() -> str:
    bin_ = _require_binary("hermes")
    # Hermes also needs a populated ~/.hermes config to know how to
    # contact a model. Refuse to run blindly against a missing one.
    if not (Path.home() / ".hermes" / "config.yaml").is_file():
        pytest.skip("hermes is installed but ~/.hermes/config.yaml is missing")
    return bin_


@pytest.fixture
def hello_proj(tmp_path: Path) -> Path:
    """Tiny non-git project with a ``hello.py`` to docstring."""
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "hello.py").write_text(
        'def hello():\n    print("Hello, world!")\n'
    )
    return proj


DOCSTRING_PROMPT = (
    "Add a one-line docstring to the hello function in hello.py "
    "describing what it does. Use your edit tool. Then stop."
)
