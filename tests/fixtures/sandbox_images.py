"""Per-agent sandbox image lifecycle: pytest fixture + CLI.

Two callers for the same logic:

* ``python tests/fixtures/sandbox_images.py`` — pre-build every
  per-agent image as a CI setup step so the test runner doesn't pay
  the cold-build cost.
* ``agentcap_image_for`` pytest fixture — same logic, on demand,
  when a test requests it.

Registered as a pytest plugin in ``tests/conftest.py`` via
``pytest_plugins``.
"""

from __future__ import annotations

import argparse
import fnmatch
import sys

import pytest

from agentcap.drivers import known_drivers
from agentcap.sandbox.podman_provisioning import (
    ensure_image, ensure_machine_running,
)


def _log(msg: str) -> None:
    sys.stderr.write(f"  [sandbox-images] {msg}\n")
    sys.stderr.flush()


def build_one(agent: str) -> str:
    ensure_machine_running(log=_log)
    return ensure_image(agent, log=_log)


def build_many(agents: list[str]) -> dict[str, str | Exception]:
    """Build each agent's image, capturing per-agent failures so CI
    surfaces the full failure set in one go."""
    out: dict[str, str | Exception] = {}
    for agent in agents:
        try:
            out[agent] = build_one(agent)
        except (FileNotFoundError, RuntimeError) as exc:
            out[agent] = exc
    return out


@pytest.fixture(scope="session")
def agentcap_image_for():
    """Factory: ``agentcap_image_for("hermes")`` ensures the
    per-agent podman image is built and current. Skips if podman
    or its machine isn't available."""
    cache: dict[str, str] = {}

    def _ensure(agent: str) -> str:
        if agent in cache:
            return cache[agent]
        try:
            tag = build_one(agent)
        except (FileNotFoundError, RuntimeError) as exc:
            pytest.skip(str(exc))
        cache[agent] = tag
        return tag

    return _ensure


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

    targets = [a for a in all_agents if fnmatch.fnmatch(a, args.pattern)]
    if not targets:
        print(
            f"no agents match pattern {args.pattern!r}; "
            f"available: {', '.join(all_agents)}",
            file=sys.stderr,
        )
        return 1

    _log(f"building: {', '.join(targets)}")
    results = build_many(targets)

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
