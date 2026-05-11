"""Linux Bubblewrap backend mounted on a buildah-built image.

A ``BwrapSandbox`` owns one persistent buildah working container for
its lifetime (typically one ``agentcap run``). Each ``run()`` enters
a fresh ``buildah unshare`` (user+mount namespace), mounts the
existing container, bwraps the agent into the rootfs, and umounts on
exit — leaving the container's OverlayFS upper layer intact so
subsequent turns see the agent's accumulated state (memories,
sessions, …). ``close()`` removes the container at end of life.

Mirrors how the Lima backend uses one VM across an entire
agentcap-run; was previously one container per ``run()`` call, which
meant agents had no state continuity across turns.

Build the image once via
:func:`agentcap.sandbox.image_provisioning.ensure_image` before
constructing this sandbox. Requires unprivileged user namespaces
(see README for Ubuntu 24.04).
"""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path


_BWRAP = "bwrap"


def _image_inspect(image: str) -> dict:
    r = subprocess.run(
        ["buildah", "inspect", image],
        capture_output=True, text=True, check=True,
    )
    return json.loads(r.stdout)


def _image_env(info: dict) -> dict[str, str]:
    """Parse the image's baked ``ENV`` directives so bwrap can re-inject
    them via ``--setenv``. ``buildah unshare`` doesn't honor image ENV
    — bwrap inherits the calling shell's env, which has none of the
    image's settings."""
    raw = (info.get("OCIv1") or {}).get("config", {}).get("Env") or []
    out: dict[str, str] = {}
    for kv in raw:
        if "=" not in kv:
            continue
        k, _, v = kv.partition("=")
        out[k] = v
    return out


def _image_entrypoint(info: dict) -> list[str]:
    """Parse the image's baked ``ENTRYPOINT`` so bwrap can prepend it
    to the agent argv. ``buildah unshare`` doesn't honor it natively
    — bwrap execs argv directly. Returns an empty list if unset."""
    ep = (info.get("OCIv1") or {}).get("config", {}).get("Entrypoint")
    return list(ep) if ep else []


def build_command(
    argv: list[str],
    *,
    container_id: str,
    writable_paths: list[Path],
    readonly_paths: list[Path] | None = None,
    deny_network: bool = False,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
) -> list[str]:
    """Assemble the ``buildah unshare bash -c '<script>'`` invocation
    that mounts an existing ``container_id`` and bwraps ``argv`` into
    it. The container is *not* removed on exit (that's BwrapSandbox's
    close() responsibility) — only the namespace-local mount is
    torn down so the next bwrap can re-mount fresh.

    ``readonly_paths`` are bind-mounted via ``--ro-bind`` (host path
    visible read-only inside the sandbox at the same path). Used for
    things like ``--skills <dir>`` where we don't want the agent
    modifying the host's source.
    """
    bwrap_rest: list[str] = [
        "--proc", "/proc",
        "--dev", "/dev",
        "--tmpfs", "/tmp",
        # /root comes from the image (drivers bootstrap any per-agent
        # ``~/.hermes/`` etc. during build). The buildah working
        # container is per-run ephemeral, so any writes the agent
        # makes there are discarded when the container is torn down —
        # no need for a tmpfs to enforce that.
        "--setenv", "HOME", "/root",
        "--setenv", "PATH",
        "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "--die-with-parent",
        "--new-session",
        "--unshare-net" if deny_network else "--share-net",
    ]
    # ``cwd`` is auto-included as a writable bind: a chdir into a
    # host path that bwrap hasn't bound just fails with ``Can't chdir``.
    bound: set[str] = set()
    all_writable = list(writable_paths)
    if cwd is not None:
        all_writable.append(Path(cwd))
    for p in all_writable:
        resolved = str(Path(p).resolve())
        if resolved in bound:
            continue
        bound.add(resolved)
        bwrap_rest.extend(["--bind", resolved, resolved])
    for p in readonly_paths or []:
        resolved = str(Path(p).resolve())
        if resolved in bound:
            continue
        bound.add(resolved)
        bwrap_rest.extend(["--ro-bind", resolved, resolved])
    if cwd is not None:
        bwrap_rest.extend(["--chdir", str(cwd)])
    for k, v in (env or {}).items():
        bwrap_rest.extend(["--setenv", k, v])

    rest_quoted = " ".join(shlex.quote(a) for a in bwrap_rest)
    agent_quoted = " ".join(shlex.quote(a) for a in argv)
    ctr_quoted = shlex.quote(container_id)

    # No `exec` before bwrap: `exec` replaces the shell, destroying
    # the trap. The trap only umounts (does NOT `buildah rm`) so the
    # container's OverlayFS upper layer survives across runs.
    script = (
        "set -u\n"
        f"trap 'buildah umount {ctr_quoted} >/dev/null 2>&1 || true' EXIT\n"
        f"mnt=$(buildah mount {ctr_quoted}) || exit 1\n"
        f"{_BWRAP} --bind \"$mnt\" / {rest_quoted} -- {agent_quoted}\n"
        "exit $?\n"
    )
    return ["buildah", "unshare", "bash", "-c", script]


class BwrapSandbox:
    """bwrap-on-image sandbox. The image holds the agent CLI + deps;
    nothing on the host is visible to the agent except paths the
    driver explicitly passes via ``writable_paths``.

    Owns one persistent buildah working container for the lifetime
    of the sandbox (typically one ``agentcap run``). Created lazily
    on first ``run()``; removed by ``close()``.
    """

    name = "bwrap"

    def __init__(
        self,
        image: str,
        *,
        env: dict[str, str] | None = None,
        readonly_paths: list[Path] | None = None,
    ) -> None:
        """``image`` is a buildah image tag, e.g.
        ``agentcap-goose:latest``. Build with
        :func:`agentcap.sandbox.image_provisioning.ensure_image`.

        ``env`` adds (or overrides) baked ENV variables for the
        lifetime of this sandbox — e.g. ``{"AGENTCAP_PROXY_URL": …}``
        for the per-agent image's entrypoint script to pick up.

        ``readonly_paths`` are bind-mounted ``--ro-bind`` on every
        ``run()`` (host path visible inside the sandbox at the same
        path, read-only). Use for things like ``--skills <dir>``
        where the agent must see the dir but shouldn't modify it.

        The image's baked ENTRYPOINT (if any) is prepended to every
        ``run()``'s argv so the image's startup wrapper (e.g.
        ``agentcap-init``) gets a chance to render config files
        from env vars before exec'ing the agent.

        Inspect (env + entrypoint) is lazy — happens on first
        ``run()`` so importing this module doesn't require buildah.
        """
        self.image = image
        self._container_id: str | None = None
        self._extra_env: dict[str, str] = dict(env or {})
        self._readonly_paths: list[Path] = list(readonly_paths or [])
        self._baked_env: dict[str, str] | None = None
        self._entrypoint: list[str] | None = None

    def _ensure_inspected(self) -> None:
        if self._baked_env is None or self._entrypoint is None:
            info = _image_inspect(self.image)
            self._baked_env = _image_env(info)
            self._baked_env.update(self._extra_env)
            self._entrypoint = _image_entrypoint(info)

    def _ensure_container(self) -> str:
        """Create the working container on first use; return its id."""
        if self._container_id is None:
            r = subprocess.run(
                ["buildah", "from", self.image],
                capture_output=True, text=True, check=True,
            )
            self._container_id = r.stdout.strip()
        return self._container_id

    def close(self) -> None:
        """Tear down the working container. Idempotent."""
        if self._container_id is None:
            return
        subprocess.run(
            ["buildah", "rm", self._container_id],
            capture_output=True, text=True, check=False,
        )
        self._container_id = None

    def __enter__(self) -> "BwrapSandbox":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def wrap(
        self,
        argv: list[str],
        *,
        writable_paths: list[Path],
        deny_network: bool = False,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> list[str]:
        self._ensure_inspected()
        # Image ENV first; caller's env overrides on key collision.
        full_env = dict(self._baked_env or {})
        if env:
            full_env.update(env)
        # Prepend the image's ENTRYPOINT so its startup wrapper
        # gets a chance to render configs from env before exec'ing
        # the agent.
        final_argv = list(self._entrypoint or []) + list(argv)
        return build_command(
            final_argv,
            container_id=self._ensure_container(),
            writable_paths=writable_paths,
            readonly_paths=self._readonly_paths,
            deny_network=deny_network,
            env=full_env,
            cwd=cwd,
        )

    def run(
        self,
        argv: list[str],
        *,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        writable_paths: list[Path] | None = None,
        deny_network: bool = False,
        timeout: float | None = None,
        check: bool = False,
    ) -> subprocess.CompletedProcess:
        wrapped = self.wrap(
            argv,
            writable_paths=writable_paths or [],
            deny_network=deny_network,
            env=env,
            cwd=cwd,
        )
        return subprocess.run(
            wrapped,
            capture_output=True, text=True,
            timeout=timeout, check=check,
        )

    # mkdtemp / write_text / read_text / rmtree return host paths
    # under ``~/.cache/agentcap/runs/``. The driver passes them in
    # ``writable_paths`` so bwrap bind-mounts them into the sandbox
    # at the same host-side path — the agent sees identical paths
    # inside and outside.

    @staticmethod
    def _runs_dir() -> Path:
        d = Path.home() / ".cache" / "agentcap" / "runs"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def mkdtemp(self, prefix: str = "agentcap-") -> str:
        return tempfile.mkdtemp(prefix=prefix, dir=str(self._runs_dir()))

    def rmtree(self, path: str) -> None:
        shutil.rmtree(path, ignore_errors=True)

    def write_text(self, path: str, content: str) -> None:
        Path(path).write_text(content)

    def read_text(self, path: str) -> str:
        return Path(path).read_text()


__all__ = ["BwrapSandbox", "build_command"]
