"""Shared fixtures.

The ``agentcap_image_for`` fixture (per-agent buildah image lifecycle)
is registered via ``pytest_plugins`` below; its body and the matching
CLI ``python tests/fixtures/sandbox_images.py`` both live in
``tests/fixtures/sandbox_images.py``.

Live driver tests need an agent binary plus a reachable OpenAI-compat
``/v1`` endpoint. Both are env-gated; missing prereqs mean the test
skips. See README ``Running tests`` for full setup.

Endpoint resolution, in order:

  1. ``AGENTCAP_TEST_LLM_URL`` set         -> use it as-is.
  2. ``llama-server`` on PATH              -> spin a session-scoped
                                              llama-server. The GGUF
                                              is fetched from HF Hub
                                              (cached) if not given.
  3. otherwise                             -> skip.

By default, the fixture fetches ``unsloth/gemma-4-E4B-it-GGUF``
(smallest tool-call-capable model on the reference rig) via
``huggingface_hub.hf_hub_download``; HF caches it in
``~/.cache/huggingface/`` so subsequent runs are instant.

Env-var overrides:

  ``AGENTCAP_TEST_LLM_URL``     OpenAI-compat /v1 base URL (skips llama-server)
  ``AGENTCAP_TEST_GGUF``        local GGUF path (skip the HF fetch)
  ``AGENTCAP_TEST_LLAMA_BIN``   llama-server path (overrides PATH lookup)
  ``AGENTCAP_TEST_NGL``         ``--n-gpu-layers`` (default 999; 0=CPU)
  ``AGENTCAP_TEST_CTX_SIZE``    llama-server ctx-size (default 8192)
  ``AGENTCAP_TEST_MODEL``       model alias agents send (default gemma-4-E4B-it)
"""

from __future__ import annotations

import http.server
import os
import shutil
import socket
import socketserver
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.request import urlopen

import pytest


pytest_plugins = ["tests.fixtures.sandbox_images"]


def _log(msg: str) -> None:
    """Write a progress line to stderr (visible with ``pytest -s``)."""
    sys.stderr.write(f"  [agentcap-test] {msg}\n")
    sys.stderr.flush()


# Default test target. ``hf_hub_download`` of the gemma-4-E4B-it Q4_K_M
# is the "click and run" path — agentcap fetches the model bytes,
# user doesn't manage GGUF files. unsloth's repo is picked because its
# filename matches the convention without a vendor-name prefix.
_DEFAULT_GGUF_REPO = "unsloth/gemma-4-E4B-it-GGUF"
_DEFAULT_GGUF_FILE = "gemma-4-E4B-it-Q4_K_M.gguf"
_DEFAULT_MODEL_ALIAS = "gemma-4-E4B-it"


def _fetch_default_gguf() -> str | None:
    """Pull the default GGUF from HF Hub. Cached in the HF default
    cache dir; first call downloads ~5GB (tqdm progress on stderr),
    subsequent calls return the cached path instantly. Returns None
    on any failure — caller treats that as 'skip live tests'."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return None
    _log(
        f"fetching default GGUF "
        f"{_DEFAULT_GGUF_REPO}/{_DEFAULT_GGUF_FILE} "
        f"(cached in ~/.cache/huggingface/ after first download)…"
    )
    try:
        return hf_hub_download(
            repo_id=_DEFAULT_GGUF_REPO,
            filename=_DEFAULT_GGUF_FILE,
        )
    except Exception as exc:
        _log(f"GGUF download failed: {exc}")
        return None


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_ready(
    url: str, timeout: float = 180.0, log=lambda msg: None,
) -> None:
    """Poll a ``/v1/models`` endpoint until it responds 200 or we
    blow ``timeout`` seconds. Tiny GGUFs load in seconds; the budget
    is generous so we don't flake on a cold weight load.

    Emits a heartbeat every ~10s so the test runner shows progress
    during a slow weight load instead of looking hung."""
    deadline = time.time() + timeout
    start = time.time()
    last_hb = start
    while time.time() < deadline:
        try:
            with urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        now = time.time()
        if now - last_hb >= 10:
            log(f"waiting for llama-server… ({int(now - start)}s elapsed)")
            last_hb = now
        time.sleep(1)
    raise RuntimeError(f"llama-server never became ready at {url}")


def _agent_reachable_host() -> str:
    """The hostname the agent (inside the sandbox) should use to
    talk to a host-side server.

    * Linux/bwrap: the namespace shares the host network, so
      ``127.0.0.1`` reaches the host.
    * macOS/Lima: the VM has its own loopback, so the host appears
      as ``host.lima.internal`` (Lima's well-known DNS alias).

    Anything else falls back to ``127.0.0.1`` and is the user's
    problem to make reachable.
    """
    import platform as _platform
    if _platform.system() == "Darwin" and shutil.which("limactl"):
        return "host.lima.internal"
    return "127.0.0.1"


@pytest.fixture(scope="session")
def live_proxy_base_url():
    """OpenAI-compat ``/v1`` URL the agent (inside the sandbox) hits.

    If ``AGENTCAP_TEST_LLM_URL`` is set, return it as-is (caller is
    responsible for the server and for making it reachable from the
    sandbox).

    Otherwise, if ``AGENTCAP_TEST_GGUF`` is set and ``llama-server``
    is available, spawn one on ``0.0.0.0:<free port>`` so the
    sandbox can connect (the agent inside a Lima VM cannot reach
    the Mac host's loopback). The URL returned uses
    :func:`_agent_reachable_host` so it works on both bwrap (where
    ``127.0.0.1`` is fine) and Lima (where the host is
    ``host.lima.internal``).
    """
    url = os.environ.get("AGENTCAP_TEST_LLM_URL")
    if url:
        yield url
        return

    # Probe common ports for an already-running llama-server before
    # spawning. Lets the user keep one server alive across many
    # `pytest` invocations (per their explicit workflow preference)
    # without having to set AGENTCAP_TEST_LLM_URL every time.
    for probe_port in (8000, 8080):
        try:
            with urlopen(
                f"http://127.0.0.1:{probe_port}/v1/models", timeout=1,
            ) as r:
                if r.status == 200:
                    _log(
                        f"reusing existing llama-server on :{probe_port}"
                    )
                    yield f"http://127.0.0.1:{probe_port}/v1"
                    return
        except Exception:
            pass

    llama = os.environ.get("AGENTCAP_TEST_LLAMA_BIN") or shutil.which(
        "llama-server"
    )
    if not llama:
        pytest.skip(
            "llama-server not on PATH; either add it to PATH or set "
            "AGENTCAP_TEST_LLAMA_BIN, OR set AGENTCAP_TEST_LLM_URL "
            "to point at an existing /v1 endpoint."
        )
    gguf = os.environ.get("AGENTCAP_TEST_GGUF") or _fetch_default_gguf()
    if not gguf:
        pytest.skip(
            "couldn't obtain a GGUF; HF fetch failed and no "
            "AGENTCAP_TEST_GGUF override set."
        )

    port = _free_port()
    ctx = os.environ.get("AGENTCAP_TEST_CTX_SIZE", "8192")
    ngl = os.environ.get("AGENTCAP_TEST_NGL", "999")
    argv = [
        llama,
        "--model", gguf,
        # 0.0.0.0 so the Lima VM can reach the host via
        # host.lima.internal; 127.0.0.1 would be loopback-only.
        "--host", "0.0.0.0",
        "--port", str(port),
        "--ctx-size", ctx,
        "--reasoning", "off",
        "--jinja",
        "--n-gpu-layers", ngl,
        # `--fit off` skips llama.cpp's `common_params_fit_impl` auto
        # parameter-fitting step. We're passing --n-gpu-layers
        # explicitly so the auto-fit is redundant, and recent llama.cpp
        # builds (b9039+) crash inside the fit step on some models
        # (gemma-4 on multi-GPU hits GGML_SCHED_MAX_SPLIT_INPUTS).
        "--fit", "off",
    ]
    log_path = "/tmp/agentcap-pytest-llama.log"
    log = open(log_path, "w")
    _log(
        f"spawning llama-server on :{port} "
        f"(gguf={Path(gguf).name}, ctx={ctx}, ngl={ngl}); "
        f"server log -> {log_path}"
    )
    proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
    # Probe on 127.0.0.1 from the host side — that's what
    # the local _wait_ready can reach.
    try:
        _wait_ready(
            f"http://127.0.0.1:{port}/v1/models",
            timeout=180,
            log=_log,
        )
        _log(f"llama-server ready at :{port}")

        # Start the in-process proxy on a free port (don't hardcode
        # 8001 — collides with whatever the user has running). Bind
        # 0.0.0.0 so the Lima VM can reach it via host.lima.internal.
        # ``sandbox_for`` propagates the resulting URL into each
        # sandbox as ``AGENTCAP_PROXY_URL``; the per-agent
        # ``agentcap-init`` substitutes that into the baked config.
        import tempfile

        from agentcap.proxy import serve_in_thread
        upstream = f"http://127.0.0.1:{port}"
        proxy_port = _free_port()
        trace_dir = tempfile.mkdtemp(prefix="agentcap-pytest-traces-")
        agent_url = (
            f"http://{_agent_reachable_host()}:{proxy_port}/v1"
        )
        _log(
            f"starting in-process proxy on 0.0.0.0:{proxy_port} "
            f"-> {upstream} (agents reach it at {agent_url})"
        )
        with serve_in_thread(
            upstream, trace_dir,
            host="0.0.0.0", port=proxy_port,
        ):
            yield agent_url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()


@pytest.fixture(scope="session")
def live_model() -> str:
    return os.environ.get("AGENTCAP_TEST_MODEL", _DEFAULT_MODEL_ALIAS)


DOCSTRING_PROMPT = (
    "Add a one-line docstring to the hello function in hello.py "
    "describing what it does. Use your edit tool. Then stop."
)


_HELLO_PY = 'def hello():\n    print("Hello, world!")\n'


@pytest.fixture(scope="session")
def sandbox_for(lima_vm_for, agentcap_image_for, live_proxy_base_url):
    """Factory: ``sandbox_for("hermes")`` returns a Sandbox keyed on
    the given agent. On macOS this is the ``agentcap-<agent>``
    LimaSandbox (the fixture ensures the VM is up first); on Linux
    it's the host BwrapSandbox mounted on the per-agent buildah image
    (the fixture ensures the image is built); on other hosts it skips.

    Sandboxes are closed at session teardown so the BwrapSandbox's
    persistent buildah working container is removed (otherwise it
    accumulates across pytest sessions).
    """
    from agentcap.sandbox import get_sandbox

    cache: dict[str, object] = {}

    def _get(agent: str):
        if agent in cache:
            return cache[agent]
        import platform as _platform
        if _platform.system() == "Darwin":
            lima_vm_for(agent)
        elif _platform.system() == "Linux":
            agentcap_image_for(agent)
        else:
            pytest.skip(
                "live tests require Linux (bwrap+buildah) or macOS (lima)"
            )
        sb = get_sandbox(
            agent=agent,
            env={"AGENTCAP_PROXY_URL": live_proxy_base_url},
        )
        cache[agent] = sb
        return sb

    yield _get
    for sb in cache.values():
        close = getattr(sb, "close", None)
        if callable(close):
            close()


@pytest.fixture
def agent_proj_for(sandbox_for):
    """Factory: ``agent_proj_for("hermes")`` ensures the
    ``hermes`` binary is installed in the sandbox, then mints a
    sandbox-side temp dir seeded with ``hello.py`` for the
    docstring task. Returns ``(sandbox, proj_path)``.

    Cleanup: ``proj_path`` is removed at the end of the test via
    ``sandbox.rmtree``.

    Skips (with ``pytest.skip``) when the agent binary isn't on the
    sandbox's PATH — capture rigs should provision the per-agent VM
    or apt-install the agent before running live tests.
    """
    created: list[tuple[object, str]] = []

    def _build(agent: str) -> tuple[object, str]:
        sb = sandbox_for(agent)
        _log(f"probing {agent!r} binary in sandbox…")
        r = sb.run(
            ["sh", "-c", f"command -v {agent}"], check=False, timeout=10,
        )
        if r.returncode != 0:
            pytest.skip(
                f"{agent!r} is not on the sandbox's PATH; provision "
                f"the agentcap-{agent} VM (or install on the Linux "
                f"host) before running live tests."
            )
        proj = sb.mkdtemp(prefix=f"agentcap-{agent}-proj-")
        sb.write_text(f"{proj}/hello.py", _HELLO_PY)
        _log(f"{agent} project: {proj}")
        created.append((sb, proj))
        return sb, proj

    yield _build
    for sb, proj in created:
        sb.rmtree(proj)


def reset_hello_py(sandbox, proj: str) -> None:
    """Reset the project's ``hello.py`` to its starting content —
    used by the retry helper in test_drivers_live so each attempt
    sees a clean slate."""
    sandbox.write_text(f"{proj}/hello.py", _HELLO_PY)


# ---------------------------------------------------------------------------
# Per-agent runtime fixtures: Lima VM on macOS, buildah image on Linux
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def lima_vm_for():
    """Factory: ``lima_vm_for("hermes")`` ensures the
    ``agentcap-hermes`` VM is running. Delegates to
    :func:`agentcap.sandbox.lima_provisioning.ensure_vm` — same
    lifecycle as ``agentcap run``. VMs touched by the fixture are
    stopped at session exit."""
    from agentcap.sandbox.lima_provisioning import ensure_vm, stop_vm

    cache: dict[str, str] = {}

    def _ensure(agent: str) -> str:
        if agent in cache:
            return cache[agent]
        try:
            vm = ensure_vm(agent, log=_log)
        except (FileNotFoundError, RuntimeError) as e:
            pytest.skip(str(e))
        cache[agent] = vm
        return vm

    yield _ensure
    for vm in cache.values():
        _log(f"stopping {vm}…")
        stop_vm(vm)




@pytest.fixture
def fake_sandbox():
    """A pass-through Sandbox stub for driver/CLI unit tests that
    don't actually exercise sandbox isolation. Lives only in tests;
    no production code depends on it."""
    import os
    import tempfile

    class _FakeSandbox:
        name = "fake"

        def wrap(self, argv, *, writable_paths, deny_network=False):
            return list(argv)

        def run(
            self, argv, *, env=None, cwd=None, writable_paths=None,
            deny_network=False, timeout=None, check=False,
        ):
            full_env = {**os.environ, **(env or {})}
            return subprocess.run(
                list(argv), env=full_env, cwd=cwd,
                capture_output=True, text=True,
                timeout=timeout, check=check,
            )

        def mkdtemp(self, prefix="agentcap-"):
            return tempfile.mkdtemp(prefix=prefix)

        def rmtree(self, path):
            shutil.rmtree(path, ignore_errors=True)

        def write_text(self, path, content):
            Path(path).write_text(content)

        def read_text(self, path):
            return Path(path).read_text()

    return _FakeSandbox()


# ---------------------------------------------------------------------------
# Mock HTTP server fixture
# ---------------------------------------------------------------------------

class _RecordingHandler(http.server.BaseHTTPRequestHandler):
    """GET-only handler that records every requested path on a class
    attribute. Reset per fixture invocation."""
    received_paths: list[str] = []

    def do_GET(self):  # noqa: N802
        type(self).received_paths.append(self.path)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"ok": true}')

    def log_message(self, *args, **kwargs):  # silence the stderr noise
        pass


@pytest.fixture
def mock_http_server():
    """Spin up a tiny in-process HTTP server on a free port for the
    duration of one test. Bound to ``0.0.0.0`` so a Lima VM can
    reach it via ``host.lima.internal`` — the Mac loopback
    ``127.0.0.1`` is not network-reachable from inside the VM.

    Yields ``(port, received_paths)``: the port the server is
    listening on, and a list (live, mutated by request handlers)
    of every path the server has been hit on. Useful for asserting
    a sandboxed subprocess actually made the call we expected.
    """
    _RecordingHandler.received_paths = []
    # Pick a free port by binding to :0 first, then handing it off.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        port = s.getsockname()[1]
    httpd = socketserver.TCPServer(("0.0.0.0", port), _RecordingHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield port, _RecordingHandler.received_paths
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)
