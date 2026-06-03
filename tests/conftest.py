"""Shared pytest fixtures.

Live tests run when prereqs are present, skip otherwise. Prereqs:

  - Agent binary present in the per-agent sandbox
    (``agentcap run --agent <name>`` once provisions it).
  - ``podman`` on PATH (the fixture pulls and runs the official
    ``ghcr.io/ggml-org/llama.cpp`` server image).
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


# Default test target. ``hf_hub_download`` of Qwen3-1.7B Q8_0 is the
# "click and run" path — agentcap fetches the model bytes, user
# doesn't manage GGUF files. Qwen3-1.7B is the smallest checkpoint
# in this family that chains read → edit reliably across the four
# drivers; ~1.7 GB downloads + loads on a CI runner in a couple of
# minutes. Semantic correctness is intentionally not graded; the
# live tests verify the wire path, not the agent's task quality.
_DEFAULT_GGUF_REPO = "Qwen/Qwen3-1.7B-GGUF"
_DEFAULT_GGUF_FILE = "Qwen3-1.7B-Q8_0.gguf"
_DEFAULT_MODEL_ALIAS = "Qwen3-1.7B"

# Official llama.cpp server image, version-pinned per llama.cpp
# commit. Override via ``AGENTCAP_TEST_LLAMA_IMAGE`` to test a
# different release. CPU-only; the GPU variants are tagged
# ``server-cuda13-*`` / ``server-vulkan-*``.
_DEFAULT_LLAMA_IMAGE = "ghcr.io/ggml-org/llama.cpp:server-b9487"


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
            log(f"waiting for llama serve… ({int(now - start)}s elapsed)")
            last_hb = now
        time.sleep(1)
    raise RuntimeError(f"llama serve never became ready at {url}")


def _agent_reachable_host() -> str:
    """Hostname the agent (inside the podman container) uses to reach
    a host-side server. Podman exposes the host gateway as
    ``host.containers.internal``."""
    return "host.containers.internal"


@pytest.fixture(scope="session")
def live_llama_url():
    """Host-side server root of the llama backend (no ``/v1`` suffix).

    For tests that spawn their own proxy on top and need a directly-
    reachable upstream. Reuses an existing ``llama serve`` on
    8000/8080, or spawns one as a podman container.
    """
    for probe_port in (8000, 8080):
        try:
            with urlopen(
                f"http://127.0.0.1:{probe_port}/v1/models", timeout=1,
            ) as r:
                if r.status == 200:
                    _log(f"reusing existing llama serve on :{probe_port}")
                    yield f"http://127.0.0.1:{probe_port}"
                    return
        except Exception:
            pass

    if not shutil.which("podman"):
        pytest.skip(
            "podman not on PATH; install with brew install podman "
            "(macOS) or apt install podman (Linux)."
        )
    # macOS: bring the podman machine up before any ``podman run`` so
    # a stopped/uninitialised machine surfaces as a clear skip with
    # an install hint, not a generic ``podman run`` failure.
    from agentcap.sandbox.podman_provisioning import ensure_machine_running
    try:
        ensure_machine_running(log=_log)
    except RuntimeError as exc:
        pytest.skip(str(exc))
    gguf = os.environ.get("AGENTCAP_TEST_GGUF") or _fetch_default_gguf()
    if not gguf:
        pytest.skip(
            "couldn't obtain a GGUF; HF fetch failed and no "
            "AGENTCAP_TEST_GGUF override set."
        )
    # HF cache stores GGUFs as symlinks into ``blobs/``; the container
    # needs the realpath's parent dir bound in.
    real_gguf = Path(gguf).resolve()
    gguf_dir = real_gguf.parent
    gguf_name = real_gguf.name

    image = os.environ.get(
        "AGENTCAP_TEST_LLAMA_IMAGE", _DEFAULT_LLAMA_IMAGE,
    )
    port = _free_port()
    ctx = os.environ.get("AGENTCAP_TEST_CTX_SIZE", "8192")
    name = f"agentcap-llama-{os.getpid()}"
    argv = [
        "podman", "run", "--rm", "-d", "--name", name,
        "-p", f"127.0.0.1:{port}:8080",
        "--mount", f"type=bind,src={gguf_dir},dst=/models,ro",
        image,
        "--model", f"/models/{gguf_name}",
        "--host", "0.0.0.0",
        "--port", "8080",
        "--ctx-size", ctx,
        "--reasoning-format", "none",
        "--jinja",
    ]
    _log(
        f"spawning llama container {name} on :{port} "
        f"(image={image}, gguf={gguf_name}, ctx={ctx})"
    )
    r = subprocess.run(argv, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        # ``podman run`` failing once the host has podman is a real
        # problem (bad flags, pull failure, permissions), not a missing
        # prereq. Fail loud so CI doesn't silently green over it.
        pytest.fail(f"podman run failed: {r.stderr.strip()}")
    try:
        _wait_ready(
            f"http://127.0.0.1:{port}/v1/models",
            timeout=180,
            log=_log,
        )
        _log(f"llama container ready at :{port}")
        yield f"http://127.0.0.1:{port}"
    finally:
        subprocess.run(
            ["podman", "rm", "-f", name],
            capture_output=True, text=True, timeout=30,
        )


@pytest.fixture(scope="session")
def live_proxy_base_url(live_llama_url):
    """Agent-side ``/v1`` URL of the in-process capture proxy.

    For tests that exercise the agent ↔ proxy ↔ llama path from
    outside.
    """
    import tempfile

    from agentcap.proxy import serve_in_thread
    proxy_port = _free_port()
    capture_dir = tempfile.mkdtemp(prefix="agentcap-pytest-captures-")
    agent_url = f"http://{_agent_reachable_host()}:{proxy_port}/v1"
    _log(
        f"starting in-process proxy on 0.0.0.0:{proxy_port} "
        f"-> {live_llama_url} (agents reach it at {agent_url})"
    )
    with serve_in_thread(
        live_llama_url, capture_dir,
        host="0.0.0.0", port=proxy_port,
    ):
        yield agent_url


@pytest.fixture(scope="session")
def live_model() -> str:
    return os.environ.get("AGENTCAP_TEST_MODEL", _DEFAULT_MODEL_ALIAS)


@pytest.fixture(scope="session")
def sandbox_for(
    agentcap_image_for, live_proxy_base_url, live_model,
):
    """Factory: ``sandbox_for("hermes")`` returns a Sandbox keyed on
    the given agent. The image fixture ensures the per-agent podman
    image is built first.

    The sandbox env is seeded with ``AGENTCAP_PROXY_URL`` *and*
    ``AGENTCAP_MODEL`` so the per-agent entrypoint can start — the
    opencode init script bails out without ``AGENTCAP_MODEL``, which
    is enough to make ``command -v opencode`` (used as a skip probe
    by ``agent_proj_for``) exit non-zero and silently skip the test.
    """
    from agentcap.sandbox import get_sandbox

    cache: dict[str, object] = {}

    def _get(agent: str):
        if agent in cache:
            return cache[agent]
        agentcap_image_for(agent)
        sb = get_sandbox(
            agent=agent,
            env={
                "AGENTCAP_PROXY_URL": live_proxy_base_url,
                "AGENTCAP_MODEL": live_model,
            },
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
    """Factory: ``agent_proj_for("hermes")`` returns
    ``(sandbox, proj_path)``. The sandbox is probed for the agent
    binary (test skips if it's missing) and a fresh empty project
    dir is minted to serve as ``cwd``.

    The dir is removed at the end of the test.
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
                f"{agent!r} is not on the sandbox's PATH; build the "
                f"agentcap-{agent} image before running live tests."
            )
        proj = sb.mkdtemp(prefix=f"agentcap-{agent}-proj-")
        _log(f"{agent} project: {proj}")
        created.append((sb, proj))
        return sb, proj

    yield _build
    for sb, proj in created:
        sb.rmtree(proj)


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
# Fake huggingface_hub.HfApi for export tests
# ---------------------------------------------------------------------------


class _FakeHfApi:
    """Captures HfApi calls so the export layer can be asserted on
    without hitting the network. Records ``create_repo`` /
    ``list_repo_files`` / ``create_commit`` for the two dataset repos
    (``-captures`` + per-agent ``-traces``), and the Collections API
    surface used by ``ensure_collection`` (``list_collections``,
    ``create_collection``, ``add_collection_item``).

    Parquet payloads are read back so tests can assert row counts +
    column sets + request_ids; bytes payloads (README.md, raw trace
    files) and string-path payloads (raw trace files committed via
    ``CommitOperationAdd(path_or_fileobj=str)``) are recorded as their
    content."""

    def __init__(self):
        self.created_repos: list[dict] = []
        self.commits: list[dict] = []
        self.collections_created: list[dict] = []
        self.collection_items: list[dict] = []
        # Default to steady-state: README already in the repo, so
        # parquet-focused tests don't see the first-push README op
        # bleed into their assertions. Tests exercising first-push
        # behaviour clear this.
        self.existing_files: list[str] = ["README.md"]

    # Back-compat single-call accessor for older tests that only
    # cared about one repo.
    @property
    def created_repo(self) -> dict | None:
        return self.created_repos[0] if self.created_repos else None

    def create_repo(self, *, repo_id, repo_type, exist_ok, private=False):
        self.created_repos.append({
            "repo_id": repo_id, "repo_type": repo_type,
            "exist_ok": exist_ok, "private": private,
        })

    def list_repo_files(self, repo_id, repo_type):
        return list(self.existing_files)

    def create_commit(self, *, repo_id, repo_type, operations, commit_message):
        import pyarrow.parquet as pq

        op_list: list[dict] = []
        for op in operations:
            entry: dict = {"path_in_repo": op.path_in_repo}
            payload = op.path_or_fileobj
            if isinstance(payload, (bytes, bytearray)):
                entry["bytes"] = bytes(payload)
            elif isinstance(payload, str) and op.path_in_repo.endswith(".parquet"):
                table = pq.read_table(payload)
                entry["n_rows"] = table.num_rows
                entry["columns"] = list(table.column_names)
                entry["request_ids"] = list(table.column("request_id").to_pylist())
            else:
                # Raw file (trace JSONL/JSON). Read bytes so tests
                # can introspect the committed payload.
                from pathlib import Path as _Path
                entry["bytes"] = _Path(payload).read_bytes() if isinstance(payload, str) else b""
            op_list.append(entry)
        self.commits.append({
            "repo_id": repo_id,
            "repo_type": repo_type,
            "commit_message": commit_message,
            "operations": op_list,
        })

    # --- Collections API ---

    def list_collections(self, *, owner=None, q=None, limit=20):
        # Idempotent ensure_collection looks for an existing one by
        # title; the fake starts empty and returns whatever was made.
        for c in self.collections_created:
            if owner and c.get("namespace") != owner:
                continue
            if q and q not in (c.get("title") or ""):
                continue
            yield _FakeCollection(c["slug"], c["title"])

    def create_collection(
        self, title, *, namespace=None, description=None,
        private=False, exists_ok=False, **_,
    ):
        slug = f"{namespace}/{title}-deadbeef" if namespace else f"{title}-deadbeef"
        record = {
            "slug": slug, "title": title, "namespace": namespace,
            "description": description, "private": private,
        }
        self.collections_created.append(record)
        return _FakeCollection(slug, title)

    def add_collection_item(
        self, *, collection_slug, item_id, item_type,
        exists_ok=False, **_,
    ):
        self.collection_items.append({
            "collection_slug": collection_slug,
            "item_id": item_id,
            "item_type": item_type,
        })


class _FakeCollection:
    __slots__ = ("slug", "title")
    def __init__(self, slug: str, title: str) -> None:
        self.slug = slug
        self.title = title


@pytest.fixture
def fake_hf_api(monkeypatch):
    fake = _FakeHfApi()
    monkeypatch.setattr("huggingface_hub.HfApi", lambda *a, **kw: fake)
    return fake


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
    duration of one test. Bound to ``0.0.0.0`` so a podman container
    can reach it via ``host.containers.internal``.

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
