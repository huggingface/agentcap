"""CLI entrypoint. See ``agentcap --help`` for the subcommand list."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from urllib.parse import urlparse

import click

from . import __version__
from .drivers import known_drivers as _known_drivers


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="agentcap")
def cli() -> None:
    """agentcap: capture LLM-agent chat-completion bytes."""


def _parse_listen(listen: str) -> tuple[str, int]:
    if ":" not in listen:
        raise click.BadParameter(
            f"--listen must be HOST:PORT, got {listen!r}", param_hint="--listen"
        )
    host, _, port_s = listen.rpartition(":")
    try:
        port = int(port_s)
    except ValueError as exc:
        raise click.BadParameter(
            f"port {port_s!r} is not an integer", param_hint="--listen"
        ) from exc
    if not (0 < port < 65536):
        raise click.BadParameter(
            f"port {port} out of range", param_hint="--listen"
        )
    return host, port


def _is_hf_router_upstream(upstream: str) -> bool:
    host = (urlparse(upstream).hostname or "").lower()
    return host == "router.huggingface.co"


def _read_hf_token_cache() -> str | None:
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    try:
        token = token_path.read_text().strip()
    except OSError:
        return None
    return token or None


_WORKSPACE_DIR = ".agentcap"


def _workspace_root() -> Path:
    return Path(os.environ.get("AGENTCAP_WORKSPACE", os.getcwd())) / _WORKSPACE_DIR


def _default_workdir(agent: str, provider_slug: str) -> Path:
    import time
    utc = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    slug = provider_slug.replace("/", "-")
    return _workspace_root() / f"{agent}-{slug}-{utc}"


def _resolve_api_key(
    *, upstream: str, explicit_api_key: str | None
) -> tuple[str | None, str | None]:
    if explicit_api_key:
        return explicit_api_key, "--api-key / AGENTCAP_API_KEY"
    if not _is_hf_router_upstream(upstream):
        return None, None

    hf_env = (os.environ.get("HF_TOKEN") or "").strip()
    if hf_env:
        return hf_env, "HF_TOKEN"

    cached = _read_hf_token_cache()
    if cached:
        return cached, "~/.cache/huggingface/token"

    return None, None


@cli.command("export")
@click.argument("targets", nargs=-1)
@click.option(
    "--all", "all_runs", is_flag=True,
    help="Export every run in the workspace (mutually exclusive with positional run-ids).",
)
@click.option(
    "--push",
    required=True,
    help="Hugging Face Dataset repo to push to: <owner>/<name>[/<subdir>] "
    "(or hf://datasets/<owner>/<name>[/<subdir>]). The repo is created "
    "if it doesn't exist; the parquet lands under data/[<subdir>/]<file>.parquet "
    "so the Hub Dataset Viewer picks it up automatically.",
)
def export_cmd(targets: tuple[str, ...], all_runs: bool, push: str) -> None:
    """Render captured runs into parquet files and push to a Dataset repo.

    ``TARGETS`` is one or more run-ids (resolved against the workspace)
    or paths to a workdir/capture-dir. Use ``--all`` to export every
    run in the workspace.
    """
    import json as _json

    from .export import detect_model, parse_dataset_uri, push_dataset

    try:
        parse_dataset_uri(push)
    except ValueError as exc:
        raise click.UsageError(str(exc))
    if all_runs and targets:
        raise click.UsageError("pass --all OR positional run-ids, not both")
    if not all_runs and not targets:
        raise click.UsageError("specify one or more run-ids/paths, or pass --all")

    workspace = _workspace_root()
    if all_runs:
        if not workspace.is_dir():
            raise click.UsageError(f"no workspace at {workspace}")
        targets = tuple(
            d.name for d in sorted(workspace.iterdir())
            if d.is_dir() and (d / "run.json").is_file()
        )
        if not targets:
            raise click.UsageError(f"no runs in {workspace}")

    def _resolve(t: str) -> tuple[Path, str | None]:
        """Return (capture_dir, agent_from_run_json) for a target."""
        # 1. run-id in the workspace.
        candidate = workspace / t
        if (candidate / "captures").is_dir():
            agent_from = None
            meta = candidate / "run.json"
            if meta.is_file():
                try:
                    agent_from = _json.loads(meta.read_text()).get("agent")
                except (OSError, _json.JSONDecodeError):
                    pass
            return candidate / "captures", agent_from
        # 2. an arbitrary workdir path with captures/ subdir.
        p = Path(t)
        if (p / "captures").is_dir():
            agent_from = None
            meta = p / "run.json"
            if meta.is_file():
                try:
                    agent_from = _json.loads(meta.read_text()).get("agent")
                except (OSError, _json.JSONDecodeError):
                    pass
            return p / "captures", agent_from
        # 3. a path that *is* a capture dir.
        if p.is_dir() and any(p.glob("*.request.json")):
            return p, None
        raise click.UsageError(f"can't resolve {t!r} to a capture dir")

    items: list[dict] = []
    for t in targets:
        cap_dir, agent = _resolve(t)
        try:
            model = detect_model(cap_dir)
        except ValueError as exc:
            raise click.UsageError(str(exc))
        if model is None:
            raise click.UsageError(
                f"{cap_dir} has no captured requests with a model field"
            )
        items.append({"capture_dir": cap_dir, "model": model, "agent": agent})
        click.echo(
            f"  [{t}] (agent={agent or '?'}, model={model})", err=True,
        )

    n_rows_list = push_dataset(items, push)
    click.echo(
        f"agentcap export: pushed {sum(n_rows_list)} rows across "
        f"{len(items)} run(s) in 1 commit -> {push}",
        err=True,
    )


@cli.command("run")
@click.option(
    "--agent",
    type=click.Choice(_known_drivers()),
    required=True,
    help="Agent driver to use.",
)
@click.option(
    "--model",
    default=None,
    help="Model id the agent uses in its outbound requests (and that "
    "the capture proxy records as the ``model`` field). Required for "
    "all drivers — hermes used to default to its own built-in id, but "
    "that made captures lie about which model was actually run.",
)
@click.option(
    "--upstream",
    required=True,
    help="Base URL of the upstream model server (e.g. http://127.0.0.1:8000).",
)
@click.option(
    "--api-key",
    "api_key",
    default=None,
    envvar="AGENTCAP_API_KEY",
    help="Bearer token forwarded to the upstream. Required for "
    "authenticated providers (HF Router, OpenAI, Together, …); leave "
    "unset for local servers that don't auth (llama-server, vLLM). "
    "Falls back to AGENTCAP_API_KEY. For HF Router only, if unset "
    "we also auto-try HF_TOKEN and ~/.cache/huggingface/token.",
)
@click.option(
    "--listen",
    default=None,
    help="HOST:PORT the in-process capture proxy binds on "
    "(default: 127.0.0.1:8001). Override if the default port "
    "collides on your host — the agent images pick the URL up via "
    "AGENTCAP_PROXY_URL.",
)
@click.option(
    "--workdir",
    default=None,
    type=click.Path(file_okay=False, dir_okay=True),
    help="Output directory for captures, session logs, and the run summary. "
    "Defaults to ``$AGENTCAP_WORKSPACE/.agentcap/<agent>-<provider>-<utc>/`` "
    "(or under cwd if AGENTCAP_WORKSPACE is unset).",
)
@click.option(
    "--workspace",
    default=None,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Host directory to expose as the agent's cwd (bind-mounted "
    "writable into the sandbox). Use this when the corpus needs the "
    "agent to see real source — e.g. a transformers git worktree for "
    "the transformers-coding-session corpus. If omitted, the agent's "
    "cwd is a fresh hermetic temp dir (recommended for any corpus "
    "that's self-contained).",
)
@click.option(
    "--skills",
    "skills_dir",
    default=None,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Host directory containing a huggingface/skills-shaped "
    "checkout (``agents/AGENTS.md`` + ``skills/<name>/SKILL.md``). "
    "Bind-mounted read-only into the sandbox; the agent's "
    "image-side entrypoint wires it into the agent's expected "
    "discovery location (``~/.hermes/skills/`` for hermes; "
    "``AGENTS.md`` + ``skills/`` symlinks in cwd for "
    "opencode/goose/pi).",
)
@click.option(
    "--tasks",
    "tasks_file",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Plain-text file with one prompt per line (# comments + blank lines ignored).",
)
@click.option(
    "--turns",
    type=int,
    default=1,
    show_default=True,
    help="Total turns per task (1 = no follow-ups).",
)
@click.option(
    "--followup",
    type=click.Choice(["continue", "templates", "synthesized"]),
    default="continue",
    show_default=True,
    help="Follow-up strategy for turns 2..N.",
)
@click.option(
    "--synth-upstream",
    default=None,
    help="Synthesizer endpoint (only used with --followup synthesized). "
    "Should bypass the capture proxy. Defaults to --upstream.",
)
@click.option(
    "--synth-model",
    default=None,
    help="Synthesizer model id (only used with --followup synthesized). "
    "Defaults to --model.",
)
@click.option(
    "--timeout",
    type=float,
    default=1200,
    show_default=True,
    help="Per-turn timeout in seconds.",
)
def run_cmd(
    agent: str,
    model: str | None,
    upstream: str,
    api_key: str | None,
    listen: str | None,
    workdir: str,
    workspace: str | None,
    skills_dir: str | None,
    tasks_file: str,
    turns: int,
    followup: str,
    synth_upstream: str | None,
    synth_model: str | None,
    timeout: float,
) -> None:
    """Drive an agent CLI through a corpus, capture, summarise."""
    import json

    from .drivers import get_driver
    from .followups import get_followup
    from .orchestrator import Orchestrator, read_tasks_txt
    from .provider import _hostname_fallback, refine_for_sub_provider
    from .proxy import (
        IN_PROCESS_PROXY_HOST,
        IN_PROCESS_PROXY_PORT,
        serve_in_thread,
    )
    from .sandbox import require_sandbox_or_die

    # Validate before touching the sandbox / workdir so a bad CLI call
    # doesn't pay the cost of building / booting the per-agent image.

    # Bind 0.0.0.0 needs a sandbox-reachable advertise host:
    # host.lima.internal on macOS/Lima, 127.0.0.1 on Linux/bwrap.
    if listen is not None:
        host, port = _parse_listen(listen)
    else:
        host, port = IN_PROCESS_PROXY_HOST, IN_PROCESS_PROXY_PORT
    agent_host = host
    if host in ("0.0.0.0", "::"):
        import platform as _platform
        import shutil as _shutil
        if _platform.system() == "Darwin" and _shutil.which("limactl"):
            agent_host = "host.lima.internal"
        else:
            agent_host = "127.0.0.1"
    proxy_url = f"http://{agent_host}:{port}/v1"

    if not model:
        raise click.UsageError(
            f"--model is required for --agent {agent}"
        )

    if followup == "synthesized":
        resolved_synth_upstream = synth_upstream or upstream
        resolved_synth_model = synth_model or model
        fu = get_followup(
            "synthesized",
            upstream=resolved_synth_upstream,
            model=resolved_synth_model,
        )
    else:
        fu = get_followup(followup)

    api_key, api_key_source = _resolve_api_key(
        upstream=upstream,
        explicit_api_key=api_key,
    )
    if api_key_source and _is_hf_router_upstream(upstream):
        click.echo(
            f"  [auth] HF Router token source={api_key_source}",
            err=True,
        )

    # --- sandbox setup: from here on, side effects.

    def _sb_log(msg: str) -> None:
        click.echo(f"  [sandbox] {msg}", err=True)

    # Hostname classification — used by the sandbox env to pick the
    # agent's credential channel (env-var auth vs no-auth) and as part
    # of the auto-generated workdir name.
    provider_slug = refine_for_sub_provider(
        _hostname_fallback(upstream), model
    )
    click.echo(f"  [provider] {provider_slug}", err=True)

    if workdir is None:
        workdir = str(_default_workdir(agent, provider_slug))
    workdir_p = Path(workdir)
    captures = workdir_p / "captures"
    sessions = workdir_p / "sessions"
    captures.mkdir(parents=True, exist_ok=True)
    sessions.mkdir(parents=True, exist_ok=True)
    click.echo(f"  [workdir] {workdir_p}", err=True)

    # First call per agent builds/boots the image; can take minutes.
    # Env vars consumed by the image entrypoint to render configs.
    sandbox_env = {
        "AGENTCAP_PROXY_URL": proxy_url,
        "AGENTCAP_MODEL": model,
        "AGENTCAP_PROVIDER": provider_slug,
    }
    if api_key:
        sandbox_env["AGENTCAP_API_KEY"] = api_key
    sandbox_ro: list[Path] = []
    if skills_dir is not None:
        skills_abs = Path(skills_dir).resolve()
        sandbox_env["AGENTCAP_SKILLS_DIR"] = str(skills_abs)
        sandbox_ro.append(skills_abs)
    sandbox = require_sandbox_or_die(
        agent=agent, command="agentcap run", log=_sb_log,
        env=sandbox_env,
        readonly_paths=sandbox_ro,
    )

    # Without --workspace, mint a hermetic temp dir so cwd-side state
    # (Hermes auto-injects AGENTS.md) doesn't leak between runs.
    if workspace is not None:
        sandbox_cwd = str(Path(workspace).resolve())
    else:
        sandbox_cwd = sandbox.mkdtemp(prefix="agentcap-run-")

    driver_kwargs: dict = {"sandbox": sandbox, "cwd": sandbox_cwd, "model": model}
    driver = get_driver(agent, **driver_kwargs)
    tasks = read_tasks_txt(tasks_file)
    if not tasks:
        raise click.UsageError(f"no tasks found in {tasks_file}")

    click.echo(
        f"agentcap run: {len(tasks)} tasks × {turns} turns through {agent} "
        f"via proxy {host}:{port} -> {upstream}",
        err=True,
    )

    def _on_event(event: str, **kw):
        click.echo(f"  [{event}] " + " ".join(f"{k}={v}" for k, v in kw.items()), err=True)

    orch = Orchestrator(driver, fu, sessions_dir=sessions, on_event=_on_event)

    try:
        with serve_in_thread(upstream, captures, host=host, port=port):
            results = orch.run_corpus(tasks, turns_per_task=turns, timeout=timeout)
    finally:
        close = getattr(driver, "close", None)
        if callable(close):
            close()
        # Tear down the sandbox's working container / VM session.
        sb_close = getattr(sandbox, "close", None)
        if callable(sb_close):
            sb_close()

    summary = {
        "agent": agent,
        "model": model,
        "provider": provider_slug,
        "upstream": upstream,
        "proxy_listen": f"{host}:{port}",
        "turns_per_task": turns,
        "followup": followup,
        "tasks": [
            {
                "task_id": r.task_id,
                "prompt": r.prompt,
                "session_id": r.session_id,
                "completed_turns": r.completed_turns,
                "turns": [
                    {
                        "turn": t.turn,
                        "returncode": t.returncode,
                        "duration_s": round(t.duration_s, 3),
                    }
                    for t in r.turns
                ],
            }
            for r in results
        ],
    }
    (workdir_p / "run.json").write_text(json.dumps(summary, indent=2))
    n_ok = sum(1 for r in results if r.completed_turns == turns)
    click.echo(
        f"agentcap run: {n_ok}/{len(results)} tasks completed all {turns} turns; "
        f"summary -> {workdir_p / 'run.json'}",
        err=True,
    )


@cli.command("ls")
@click.option(
    "--long", "-l", "long_form", is_flag=True,
    help="Long form: include upstream and per-run task counts.",
)
def ls_cmd(long_form: bool) -> None:
    """List runs in the workspace (``$AGENTCAP_WORKSPACE/.agentcap/`` or
    ``./.agentcap/`` if AGENTCAP_WORKSPACE is unset)."""
    import json as _json

    root = _workspace_root()
    if not root.is_dir():
        click.echo(f"no workspace at {root}; run `agentcap run` first.", err=True)
        return

    rows: list[dict] = []
    for run_dir in sorted(root.iterdir()):
        meta_path = run_dir / "run.json"
        if not run_dir.is_dir() or not meta_path.is_file():
            continue
        try:
            meta = _json.loads(meta_path.read_text())
        except (OSError, _json.JSONDecodeError):
            continue
        captures = run_dir / "captures"
        n_caps = (
            len(list(captures.glob("*.request.json"))) if captures.is_dir() else 0
        )
        tasks = meta.get("tasks") or []
        turns = meta.get("turns_per_task", 1)
        n_ok = sum(1 for t in tasks if t.get("completed_turns") == turns)
        rows.append({
            "run_id": run_dir.name,
            "agent": meta.get("agent") or "?",
            "model": (meta.get("model") or "?").split("/")[-1],
            "provider": meta.get("provider") or "?",
            "upstream": meta.get("upstream") or "?",
            "n_tasks": len(tasks),
            "n_ok": n_ok,
            "n_caps": n_caps,
        })

    if not rows:
        click.echo(f"no runs in {root}.", err=True)
        return

    if long_form:
        cols = ["run_id", "agent", "model", "provider", "tasks", "captures", "upstream"]
        widths = [
            max(len("run_id"), max(len(r["run_id"]) for r in rows)),
            max(len("agent"), max(len(r["agent"]) for r in rows)),
            max(len("model"), max(len(r["model"]) for r in rows)),
            max(len("provider"), max(len(r["provider"]) for r in rows)),
            len("tasks"),
            len("captures"),
            max(len("upstream"), max(len(r["upstream"]) for r in rows)),
        ]
    else:
        cols = ["run_id", "agent", "model", "tasks", "captures"]
        widths = [
            max(len("run_id"), max(len(r["run_id"]) for r in rows)),
            max(len("agent"), max(len(r["agent"]) for r in rows)),
            max(len("model"), max(len(r["model"]) for r in rows)),
            len("tasks"),
            len("captures"),
        ]

    def _fmt(cells: list[str]) -> str:
        return "  ".join(c.ljust(w) for c, w in zip(cells, widths))

    click.echo(_fmt([c.upper() for c in cols]))
    for r in rows:
        tasks_cell = f"{r['n_ok']}/{r['n_tasks']}"
        if long_form:
            click.echo(_fmt([
                r["run_id"], r["agent"], r["model"], r["provider"],
                tasks_cell, str(r["n_caps"]), r["upstream"],
            ]))
        else:
            click.echo(_fmt([
                r["run_id"], r["agent"], r["model"],
                tasks_cell, str(r["n_caps"]),
            ]))


def main() -> int:
    cli.main(standalone_mode=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
