"""Per-agent sandbox image lifecycle: pytest fixtures + CLI.

Two backend-specific fixtures expose the same shape:

* ``agentcap_buildah_image_for`` — buildah-built image for the bwrap
  end-to-end tests.
* ``agentcap_podman_image_for`` — podman-built image for the podman
  end-to-end tests and the default ``sandbox_for`` factory.

A test binds to whichever fixture matches its backend; ``sandbox_for``
dispatches to the right one based on the autodetected backend.

Registered as a pytest plugin in ``tests/conftest.py`` via
``pytest_plugins``.
"""

from __future__ import annotations

import argparse
import fnmatch
import sys

import pytest

from agentcap.drivers import known_drivers
from agentcap.sandbox import _autodetect_backend


def _log(msg: str) -> None:
    sys.stderr.write(f"  [sandbox-images] {msg}\n")
    sys.stderr.flush()


def _ensure_buildah_image(agent: str) -> str:
    from agentcap.sandbox.image_provisioning import ensure_image
    return ensure_image(agent, log=_log)


def _ensure_podman_image(agent: str) -> str:
    from agentcap.sandbox.podman_provisioning import (
        ensure_image, ensure_machine_running,
    )
    ensure_machine_running(log=_log)
    return ensure_image(agent, log=_log)


def build_many(agents: list[str], builder) -> dict[str, str | Exception]:
    """Run ``builder(agent)`` for each agent, capturing per-agent
    failures so CI surfaces the full failure set in one go."""
    out: dict[str, str | Exception] = {}
    for agent in agents:
        try:
            out[agent] = builder(agent)
        except (FileNotFoundError, RuntimeError) as exc:
            out[agent] = exc
    return out


def _image_fixture(builder):
    """Wrap a per-backend image builder in a session-scoped cache +
    pytest.skip on missing tooling."""
    cache: dict[str, str] = {}

    def _ensure(agent: str) -> str:
        if agent in cache:
            return cache[agent]
        try:
            tag = builder(agent)
        except (FileNotFoundError, RuntimeError) as exc:
            pytest.skip(str(exc))
        cache[agent] = tag
        return tag

    return _ensure


@pytest.fixture(scope="session")
def agentcap_buildah_image_for():
    """Factory for the bwrap end-to-end tests."""
    return _image_fixture(_ensure_buildah_image)


@pytest.fixture(scope="session")
def agentcap_podman_image_for():
    """Factory for the podman end-to-end tests."""
    return _image_fixture(_ensure_podman_image)


@pytest.fixture(scope="session")
def agentcap_image_for():
    """Factory that follows the autodetected backend — buildah for
    bwrap, podman otherwise. For tests that exercise whatever
    ``agentcap run`` itself would use."""
    builder = (
        _ensure_buildah_image
        if _autodetect_backend() == "bwrap"
        else _ensure_podman_image
    )
    return _image_fixture(builder)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Pre-build the per-agent sandbox images used by "
            "`agentcap run` and the live driver tests."
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available agent names and exit.",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "buildah", "podman"),
        default="auto",
        help=(
            "Image-builder backend. ``auto`` follows AGENTCAP_SANDBOX "
            "/ the OS default (buildah for bwrap, podman otherwise)."
        ),
    )
    parser.add_argument(
        "pattern",
        nargs="?",
        default="*",
        help=(
            "Glob pattern to filter agents (e.g. 'goose', 'pi', "
            "'*'). Default: '*' (all)."
        ),
    )
    args = parser.parse_args()

    all_agents = sorted(known_drivers())

    if args.list:
        for name in all_agents:
            print(name)
        return 0

    if args.backend == "auto":
        builder_kind = "buildah" if _autodetect_backend() == "bwrap" else "podman"
    else:
        builder_kind = args.backend
    builder = _ensure_buildah_image if builder_kind == "buildah" else _ensure_podman_image

    targets = [a for a in all_agents if fnmatch.fnmatch(a, args.pattern)]
    if not targets:
        print(
            f"no agents match pattern {args.pattern!r}; "
            f"available: {', '.join(all_agents)}",
            file=sys.stderr,
        )
        return 1

    _log(f"building ({builder_kind}): {', '.join(targets)}")
    results = build_many(targets, builder)

    ok = {a: t for a, t in results.items() if not isinstance(t, Exception)}
    failed = {a: e for a, e in results.items() if isinstance(e, Exception)}

    for agent, tag in ok.items():
        _log(f"  OK   {agent} -> {tag}")
    for agent, exc in failed.items():
        _log(f"  FAIL {agent}: {exc}")

    if failed:
        _log(f"{len(failed)}/{len(targets)} failed")
        return 1
    _log(f"all {len(targets)} images ready")
    return 0


if __name__ == "__main__":
    sys.exit(main())
