"""CLI entrypoint. See ``agentcap --help`` for the subcommand list."""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import urlparse

import click

from . import __version__
from .drivers import known_drivers as _known_drivers


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="agentcap")
def cli() -> None:
    """agentcap: capture LLM-agent chat-completion bytes."""


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


def _workspace_source() -> tuple[Path, str]:
    """Return the workspace root (without ``.agentcap`` suffix) and a
    short label of where the value came from. Used by error messages
    so the user can see, verbatim, what AGENTCAP_WORKSPACE resolved
    to — catches shell typos like ``WORKSPACE==/path`` that leave a
    leading ``=`` in the env var value."""
    env = os.environ.get("AGENTCAP_WORKSPACE")
    if env is not None:
        return Path(env), f"AGENTCAP_WORKSPACE={env!r}"
    return Path(os.getcwd()), "cwd (AGENTCAP_WORKSPACE unset)"


def _workspace_root() -> Path:
    base, _ = _workspace_source()
    return base / _WORKSPACE_DIR


def _no_workspace_msg(workspace: Path) -> str:
    _, src = _workspace_source()
    return (
        f"no workspace at {str(workspace)!r} (from {src}). "
        f"Run `agentcap run` first, or set AGENTCAP_WORKSPACE to a "
        f"directory that contains a ``.agentcap/`` subdir."
    )


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


def _complete_run_ids(ctx, param, incomplete):
    """Shell completion for workspace run-ids."""
    root = _workspace_root()
    if not root.is_dir():
        return []
    return [
        d.name for d in root.iterdir()
        if d.is_dir() and (d / "run.json").is_file()
        and d.name.startswith(incomplete)
    ]


def _complete_request_ids(ctx, param, incomplete):
    """Shell completion for captured request-ids across the workspace."""
    root = _workspace_root()
    if not root.is_dir():
        return []
    out: list[str] = []
    for run_dir in root.iterdir():
        captures = run_dir / "captures"
        if not captures.is_dir():
            continue
        for req in captures.glob(f"{incomplete}*.request.json"):
            out.append(req.name.removesuffix(".request.json"))
    return out


@cli.command("export")
@click.argument("targets", nargs=-1, shell_complete=_complete_run_ids)
@click.option(
    "--all", "all_runs", is_flag=True,
    help="Export every run in the workspace (mutually exclusive with positional run-ids).",
)
@click.option(
    "--push",
    required=True,
    help="``<owner>/<base>`` — the Hugging Face Collection base name. "
    "Captures parquets land in ``<owner>/<base>-captures``, raw session "
    "traces in ``<owner>/<base>-traces``, and both are added to a "
    "Collection titled ``<base>`` under ``<owner>``. Repos and "
    "Collection are created on first push.",
)
@click.option(
    "--no-scan", "no_scan", is_flag=True,
    help="Skip the pre-export trufflehog secret scan. Off by default: "
    "any **verified** secret in a target run dir aborts the export "
    "before any push happens.",
)
def export_cmd(
    targets: tuple[str, ...], all_runs: bool, push: str, no_scan: bool,
) -> None:
    """Render captured runs into parquets + upload native traces, in
    one shot. Pushes to a paired ``-captures``/``-traces`` dataset
    grouped under a Collection. ``TARGETS`` is one or more run-ids
    (resolved against the workspace) or paths to a workdir; ``--all``
    exports every run in the workspace.
    """
    import json as _json

    from .export import (
        captures_repo_id,
        detect_model,
        ensure_collection,
        parse_collection_base,
        push_agent_traces_dataset,
        push_captures_dataset,
    )

    try:
        owner, base = parse_collection_base(push)
    except ValueError as exc:
        raise click.UsageError(str(exc))
    if all_runs and targets:
        raise click.UsageError("pass --all OR positional run-ids, not both")
    if not all_runs and not targets:
        raise click.UsageError("specify one or more run-ids/paths, or pass --all")

    workspace = _workspace_root()
    if all_runs:
        if not workspace.is_dir():
            raise click.UsageError(_no_workspace_msg(workspace))
        targets = tuple(
            d.name for d in sorted(workspace.iterdir())
            if d.is_dir() and (d / "run.json").is_file()
        )
        if not targets:
            raise click.UsageError(f"no runs in {workspace}")

    def _resolve(t: str) -> tuple[Path, str | None, str]:
        """Return (capture_dir, agent_from_run_json, run_id) for a target.
        run_id is the run-dir basename; it labels both the captures
        rows and the traces-dataset folder."""
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
            return candidate / "captures", agent_from, candidate.name
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
            return p / "captures", agent_from, p.name
        # 3. a path that *is* a capture dir.
        if p.is_dir() and any(p.glob("*.request.json")):
            return p, None, p.parent.name
        raise click.UsageError(f"can't resolve {t!r} to a capture dir")

    cap_items: list[dict] = []
    trace_items: list[dict] = []
    for t in targets:
        cap_dir, agent, run_id = _resolve(t)
        try:
            model = detect_model(cap_dir)
        except ValueError as exc:
            raise click.UsageError(str(exc))
        if model is None:
            if all_runs:
                click.echo(f"  [{t}] skipped (no captures)", err=True)
                continue
            raise click.UsageError(
                f"{cap_dir} has no captured requests with a model field"
            )
        cap_items.append({
            "capture_dir": cap_dir, "model": model, "agent": agent,
            "run_id": run_id,
        })
        # Traces dir is sibling to captures; missing/empty is fine —
        # push_traces_dataset accepts it and just records 0 files.
        traces_dir = cap_dir.parent / "traces"
        trace_items.append({"traces_dir": traces_dir, "run_id": run_id})
        n_traces = sum(1 for _ in traces_dir.iterdir()) \
            if traces_dir.is_dir() else 0
        click.echo(
            f"  [{t}] (agent={agent or '?'}, model={model}, "
            f"traces={n_traces})",
            err=True,
        )
    if not cap_items:
        raise click.UsageError("no runs with captures to export")

    # Pre-export gate: refuse to push if any run carries a verified
    # secret. Verification round-trips to each provider's API so
    # ``verified`` is high-precision (real, live credential).
    # Unverified hits are surfaced but don't block — pattern-only
    # detectors hit a real false-positive rate on model output.
    if not no_scan:
        run_dirs = [Path(c["capture_dir"]).parent for c in cap_items]
        n_verified = _scan_run_dirs(run_dirs, no_verification=False)
        if n_verified > 0:
            raise click.ClickException(
                f"export aborted: trufflehog found {n_verified} verified "
                "secret(s) — see output above. Inspect, redact, or pass "
                "--no-scan to override."
            )

    cap_repo, n_rows_list = push_captures_dataset(
        cap_items, owner=owner, base=base,
    )
    click.echo(
        f"agentcap export: pushed {sum(n_rows_list)} rows across "
        f"{len(cap_items)} run(s) -> {cap_repo}",
        err=True,
    )

    # Group traces by agent — one dataset per agent so the Hub
    # viewer doesn't try to merge incompatible schemas.
    by_agent: dict[str, list[dict]] = {}
    for cap, tr in zip(cap_items, trace_items):
        agent_name = cap.get("agent") or "unknown"
        n = sum(1 for _ in tr["traces_dir"].iterdir()) \
            if tr["traces_dir"].is_dir() else 0
        if n == 0:
            continue
        by_agent.setdefault(agent_name, []).append(tr)

    traces_repos: list[str] = []
    for agent_name, tr_items in sorted(by_agent.items()):
        tr_repo, n_files = push_agent_traces_dataset(
            tr_items, owner=owner, base=base, agent=agent_name,
        )
        traces_repos.append(tr_repo)
        click.echo(
            f"agentcap export: pushed {n_files} trace file(s) for "
            f"{agent_name} across {len(tr_items)} run(s) -> {tr_repo}",
            err=True,
        )

    slug = ensure_collection(
        owner=owner, base=base,
        repos=[captures_repo_id(owner, base), *traces_repos],
    )
    click.echo(
        f"agentcap export: collection -> https://huggingface.co/collections/{slug}",
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
    "unset for local servers that don't auth (llama serve, vLLM). "
    "Falls back to AGENTCAP_API_KEY. For HF Router only, if unset "
    "we also auto-try HF_TOKEN and ~/.cache/huggingface/token.",
)
@click.option(
    "--sandbox",
    "sandbox_dir",
    default=None,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Host directory exposed as the agent's cwd (bind-mounted "
    "writable into the per-agent container). Use this when the corpus "
    "needs the agent to see real source — e.g. a transformers git "
    "worktree for the transformers-coding-session corpus. If omitted, "
    "an empty ``sandbox/`` is created next to ``captures/`` under the "
    "auto-derived run dir.",
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
    sandbox_dir: str | None,
    skills_dir: str | None,
    tasks_file: str,
    turns: int,
    followup: str,
    timeout: float,
) -> None:
    """Drive an agent CLI through a corpus, capture, summarise."""
    import json

    from .drivers import get_driver, traces_dump_argv_for
    from .followups import get_followup
    from .orchestrator import Orchestrator, read_tasks_txt
    from .provider import _hostname_fallback, refine_for_sub_provider
    from .proxy import serve_in_thread
    from .sandbox import require_sandbox_or_die

    if not model:
        raise click.UsageError(
            f"--model is required for --agent {agent}"
        )

    api_key, api_key_source = _resolve_api_key(
        upstream=upstream,
        explicit_api_key=api_key,
    )
    if api_key_source and _is_hf_router_upstream(upstream):
        click.echo(
            f"  [auth] HF Router token source={api_key_source}",
            err=True,
        )

    if followup == "synthesized":
        fu = get_followup(
            "synthesized", upstream=upstream, model=model, api_key=api_key
        )
    else:
        fu = get_followup(followup)

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

    workdir_p = _default_workdir(agent, provider_slug)
    captures = workdir_p / "captures"
    sessions = workdir_p / "sessions"
    traces = workdir_p / "traces"
    state = workdir_p / "state"
    captures.mkdir(parents=True, exist_ok=True)
    sessions.mkdir(parents=True, exist_ok=True)
    traces.mkdir(parents=True, exist_ok=True)
    state.mkdir(parents=True, exist_ok=True)
    click.echo(f"  [workdir] {workdir_p}", err=True)

    # Stub run.json so ``agentcap ls/inspect/export`` can discover this
    # run while it's still in flight. Fully overwritten with the final
    # summary (incl. per-task durations) at end-of-run.
    (workdir_p / "run.json").write_text(json.dumps({
        "agent": agent,
        "model": model,
        "provider": provider_slug,
        "upstream": upstream,
        "turns_per_task": turns,
        "followup": followup,
        "tasks": [],
    }, indent=2))

    # Resolve --sandbox up front: it joins the bind-mount set
    # alongside --skills (RO) and the traces dir (RW).
    if sandbox_dir is not None:
        sandbox_cwd = str(Path(sandbox_dir).resolve())
    else:
        default_sandbox = workdir_p / "sandbox"
        default_sandbox.mkdir(parents=True, exist_ok=True)
        sandbox_cwd = str(default_sandbox)

    tasks = read_tasks_txt(tasks_file)
    if not tasks:
        raise click.UsageError(f"no tasks found in {tasks_file}")

    def _on_event(event: str, **kw):
        click.echo(f"  [{event}] " + " ".join(f"{k}={v}" for k, v in kw.items()), err=True)

    # Bind on 0.0.0.0 so the podman container (which has its own netns)
    # can dial in via ``host.containers.internal``. Loopback would be
    # unreachable from the container side.
    with serve_in_thread(upstream, captures, host="0.0.0.0") as proxy:
        proxy_url = f"http://host.containers.internal:{proxy.port}/v1"
        click.echo(f"  [proxy] {proxy_url}", err=True)

        sandbox_env = {
            "AGENTCAP_PROXY_URL": proxy_url,
            "AGENTCAP_MODEL": model,
            "AGENTCAP_PROVIDER": provider_slug,
            "AGENTCAP_TRACES_DIR": str(traces.resolve()),
            # State dir: SQLite-backed agents (hermes, goose, opencode)
            # redirect their session store at it, so the .db lands on
            # host as it's written — survives container crashes. Pi
            # streams JSONL via the traces symlink and ignores this.
            "AGENTCAP_STATE_DIR": str(state.resolve()),
        }
        if api_key:
            sandbox_env["AGENTCAP_API_KEY"] = api_key
        sandbox_ro: list[Path] = []
        if skills_dir is not None:
            skills_abs = Path(skills_dir).resolve()
            sandbox_env["AGENTCAP_SKILLS_DIR"] = str(skills_abs)
            sandbox_ro.append(skills_abs)
        sandbox_rw: list[Path] = [
            traces.resolve(),
            state.resolve(),
            Path(sandbox_cwd).resolve(),
        ]
        # First call per agent builds/boots the image; can take minutes.
        sandbox = require_sandbox_or_die(
            agent=agent, command="agentcap run", log=_sb_log,
            env=sandbox_env,
            readonly_paths=sandbox_ro,
            writable_paths=sandbox_rw,
        )

        driver_kwargs: dict = {
            "sandbox": sandbox, "cwd": sandbox_cwd, "model": model,
        }
        driver = get_driver(agent, **driver_kwargs)

        click.echo(
            f"agentcap run: {len(tasks)} tasks × {turns} turns through "
            f"{agent} -> {upstream}",
            err=True,
        )
        orch = Orchestrator(
            driver, fu, sessions_dir=sessions, on_event=_on_event,
            set_capture_context=proxy.set_context,
        )

        try:
            results = orch.run_corpus(
                tasks, turns_per_task=turns, timeout=timeout,
            )
        finally:
            # Dump SQLite-stored sessions to AGENTCAP_TRACES_DIR for
            # agents whose images ship a ``dump-traces`` script
            # (goose, opencode). No-op for symlink-style agents
            # (pi, hermes) — their transcripts already streamed to
            # the host. Failure is logged but never aborts the run.
            dump_argv = traces_dump_argv_for(agent)
            if dump_argv is not None:
                try:
                    r = sandbox.run(
                        dump_argv,
                        env=sandbox_env,
                        cwd=sandbox_cwd,
                        timeout=600,
                    )
                    if r.returncode != 0:
                        click.echo(
                            f"  [traces] dump-traces rc={r.returncode}",
                            err=True,
                        )
                except Exception as exc:
                    click.echo(f"  [traces] dump-traces failed: {exc}", err=True)
            close = getattr(driver, "close", None)
            if callable(close):
                close()
            sb_close = getattr(sandbox, "close", None)
            if callable(sb_close):
                sb_close()

    summary = {
        "agent": agent,
        "model": model,
        "provider": provider_slug,
        "upstream": upstream,
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


def _scan_run_dirs(
    run_dirs: list[Path],
    *,
    no_verification: bool = False,
    rescan: bool = False,
) -> int:
    """Run trufflehog over each run dir; print a per-run summary.
    Returns the total count of **verified** hits across all runs.
    Unverified hits are listed but never abort the caller —
    Trufflehog's pattern matchers have a real false-positive rate.

    Persists results to ``<run_dir>/scan.json`` so repeat scans skip
    the verification round-trips. Pass ``rescan=True`` to force a
    fresh scan."""
    from collections import Counter

    from .scan import TrufflehogMissingError, scan_run_dir

    total_verified = 0
    for run_dir in run_dirs:
        try:
            result, was_cached = scan_run_dir(
                run_dir,
                no_verification=no_verification,
                rescan=rescan,
            )
        except TrufflehogMissingError as exc:
            raise click.ClickException(str(exc))
        n_unver = len(result.unverified)
        n_ver = len(result.verified)
        total_verified += n_ver
        cache_tag = " (cached)" if was_cached else ""
        click.echo(
            f"  [scan] {run_dir.name}{cache_tag}: "
            f"{result.chunks_scanned} chunks / {result.bytes_scanned} bytes; "
            f"verified={n_ver} unverified={n_unver}",
            err=True,
        )
        # Verified hits are rare + actionable — list each one.
        for hit in result.verified:
            click.echo(
                f"    VERIFIED  {hit.detector}  {hit.file}",
                err=True,
            )
        # Unverified hits are usually pattern-only false positives
        # (Box matches any 32-char alphanumeric, Mailgun any 32-hex,
        # …). Summarise by detector instead of dumping every line;
        # per-hit detail lives in ``<run_dir>/scan.json``.
        if result.unverified:
            by_det = Counter(h.detector for h in result.unverified)
            tail = ", ".join(
                f"{det}={n}" for det, n in by_det.most_common()
            )
            click.echo(f"    unverified by detector: {tail}", err=True)
    return total_verified


@cli.command("scan")
@click.argument("targets", nargs=-1)
@click.option(
    "--all", "all_runs", is_flag=True,
    help="Scan every run in the workspace.",
)
@click.option(
    "--no-verify", "no_verify", is_flag=True,
    help="Skip the provider-API verification step. Faster + offline-"
    "safe, but every hit lands as unverified so the gate never fires.",
)
@click.option(
    "--rescan", is_flag=True,
    help="Ignore any cached ``<run_dir>/scan.json`` and re-run trufflehog.",
)
def scan_cmd(
    targets: tuple[str, ...], all_runs: bool, no_verify: bool, rescan: bool,
) -> None:
    """Run trufflehog over a run's captures+traces. Exits non-zero on
    any verified hit; unverified hits are listed but don't change the
    exit code (false-positive rate is real). Verification is on by
    default — disable with --no-verify. Results are persisted to
    ``<run_dir>/scan.json`` and reused on subsequent runs unless
    --rescan is passed."""
    workspace = _workspace_root()
    if all_runs and targets:
        raise click.UsageError("pass --all OR positional run-ids, not both")
    if not all_runs and not targets:
        raise click.UsageError("specify one or more run-ids/paths, or pass --all")

    if all_runs:
        if not workspace.is_dir():
            raise click.UsageError(_no_workspace_msg(workspace))
        run_dirs = [
            d for d in sorted(workspace.iterdir())
            if d.is_dir() and (d / "run.json").is_file()
        ]
        if not run_dirs:
            raise click.UsageError(f"no runs in {workspace}")
    else:
        run_dirs = []
        for t in targets:
            cand = workspace / t
            if (cand / "run.json").is_file():
                run_dirs.append(cand)
                continue
            p = Path(t)
            if (p / "run.json").is_file():
                run_dirs.append(p)
                continue
            raise click.UsageError(f"can't resolve {t!r} to a run dir")

    n_verified = _scan_run_dirs(
        run_dirs, no_verification=no_verify, rescan=rescan,
    )
    if n_verified > 0:
        raise click.ClickException(
            f"scan found {n_verified} verified secret(s); see output above"
        )
    click.echo("scan: no verified secrets found.", err=True)


@cli.command("ls")
@click.argument(
    "workspace",
    required=False,
    type=click.Path(file_okay=False, dir_okay=True, resolve_path=False),
)
@click.option(
    "--long", "-l", "long_form", is_flag=True,
    help="Long form: include upstream and per-run task counts.",
)
def ls_cmd(workspace: str | None, long_form: bool) -> None:
    """List runs under a local workspace.

    Without ``WORKSPACE``, looks at ``./.agentcap/``. Accepts either
    the parent dir (where ``agentcap run`` created the ``.agentcap/``
    subdir) or the ``.agentcap/`` dir itself.

    Unlike ``agentcap run`` / ``export``, ``ls`` does NOT consult
    ``$AGENTCAP_WORKSPACE`` — what you point it at is what you get.
    """
    import json as _json

    if workspace is None:
        root = Path.cwd() / _WORKSPACE_DIR
    else:
        # Normalize before checking .name so paths like ``.``,
        # ``.agentcap/.`` or ``foo/`` classify correctly (``Path('.').name``
        # is ``''``, not ``'.agentcap'``).
        p = Path(os.path.normpath(workspace)).absolute()
        root = p if p.name == _WORKSPACE_DIR else p / _WORKSPACE_DIR
    if not root.is_dir():
        click.echo(
            f"no workspace at {str(root)!r}. "
            f"Run `agentcap run` first, or pass a directory that "
            f"contains a ``.agentcap/`` subdir.",
            err=True,
        )
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


def _resolve_request_id(
    rid: str, source: str | None
) -> tuple[str, dict, dict | None, dict | None, Path | None]:
    """Resolve ``rid`` (full or short prefix) to
    ``(full_rid, body, response_record, request_record, capture_dir)``.

    - If ``source`` is given, looks the rid up there via
      ``replay.load_request`` (any agentcap-supported source: dir,
      parquet, hf://) — exact match only. Response and request
      records and ``capture_dir`` are unavailable in that path
      (just the body).
    - Otherwise scans the workspace, accepting a prefix (git-style)
      and returning the body, the paired response, the full
      request record (which carries ``task_id``, ``turn``,
      ``captured_at``, ``upstream_url``), and the capture dir the
      rid was found in — exposing the dir lets callers load
      sibling captures without scanning again.
    """
    from . import replay

    if source is not None:
        try:
            return rid, replay.load_request(source, rid), None, None, None
        except KeyError as exc:
            raise click.UsageError(str(exc))
        except (ValueError, FileNotFoundError) as exc:
            raise click.UsageError(str(exc))

    try:
        found = replay.resolve_workspace_rid(_workspace_root(), rid)
    except replay.AmbiguousRequestId as exc:
        raise click.UsageError(str(exc))
    if found is None:
        raise click.UsageError(
            f"request_id {rid!r} not found in workspace at {_workspace_root()}; "
            f"pass --source to point at a capture dir, parquet, or hf:// URI."
        )
    capture_dir, full_rid = found
    import json as _json
    req_rec = _json.loads(
        (capture_dir / f"{full_rid}.request.json").read_text()
    )
    resp_path = capture_dir / f"{full_rid}.response.json"
    resp_rec = (
        _json.loads(resp_path.read_text()) if resp_path.is_file() else None
    )
    body = req_rec.get("body")
    if not isinstance(body, dict):
        raise click.UsageError(
            f"capture {capture_dir / f'{full_rid}.request.json'} has no body field"
        )
    return full_rid, body, resp_rec, req_rec, capture_dir


def _enumerate_workspace_requests(scope: str | None) -> list[dict]:
    """Walk captures across the workspace (or one run if ``scope`` is a
    run-id) and return one row per captured request, grouped by run
    then chronological within each run. Each row has ``run_id``,
    ``rid``, ``captured_at``, ``status``, and ``preview`` (last user
    message, truncated)."""
    import json as _json

    root = _workspace_root()
    if not root.is_dir():
        return []
    run_dirs = (
        [root / scope] if scope else [d for d in sorted(root.iterdir()) if d.is_dir()]
    )
    rows: list[dict] = []
    for run_dir in run_dirs:
        captures = run_dir / "captures"
        if not captures.is_dir():
            continue
        # Sort within (task, time) so per-task ``prev_rid`` is the
        # immediately-preceding capture in chronological order.
        recs: list[tuple[str, dict]] = []
        for req_path in captures.glob("*.request.json"):
            rid = req_path.stem.split(".")[0]
            try:
                req = _json.loads(req_path.read_text())
            except (OSError, _json.JSONDecodeError):
                continue
            recs.append((rid, req))
        recs.sort(
            key=lambda r: (r[1].get("task_id") or "", r[1].get("captured_at", 0))
        )
        prev_rid_by_task: dict = {}
        prev_msgs_by_task: dict = {}
        idx_by_task: dict = {}
        for rid, req in recs:
            resp_path = captures / f"{rid}.response.json"
            status = "?"
            if resp_path.is_file():
                try:
                    status = str(_json.loads(resp_path.read_text()).get("status_code", "?"))
                except (OSError, _json.JSONDecodeError):
                    pass
            messages = (req.get("body") or {}).get("messages") or []
            task_id = req.get("task_id")
            # When task_id is missing, key the per-task caches on the
            # rid so unrelated orphan captures don't accidentally chain
            # together for the diff / prev_rid / req_index.
            task_key = task_id if task_id is not None else rid
            prev_msgs = prev_msgs_by_task.get(task_key)
            if prev_msgs is None:
                new_msgs = messages
                label = f"(init {len(new_msgs)})"
            else:
                removed, new_msgs = _diff_messages(prev_msgs, messages)
                label = f"({_delta_label(len(removed), len(new_msgs))})"
            summary = _message_summary(new_msgs[-1]) if new_msgs else ""
            preview = f"{label} {summary}".replace("\n", " ").strip()
            # Concatenate every new message's content into a single
            # searchable blob so fzf can match against deeper content
            # (e.g. ``hf-cli`` referenced 4 messages back in the diff)
            # without bloating the visible row.
            searchable = " ".join(
                _message_text(m) for m in new_msgs
            ).replace("\n", " ").replace("\t", " ")
            prev_rid = prev_rid_by_task.get(task_key)
            prev_msgs_by_task[task_key] = messages
            prev_rid_by_task[task_key] = rid
            idx_by_task[task_key] = idx_by_task.get(task_key, 0) + 1
            rows.append({
                "run_id": run_dir.name,
                "rid": rid,
                "captured_at": int(req.get("captured_at", 0)),
                "status": status,
                "task_id": task_id,
                "turn": req.get("turn"),
                "req_index": idx_by_task[task_key],
                "prev_rid": prev_rid,
                "preview": preview,
                "searchable": searchable,
            })
    rows.sort(key=lambda r: (r["run_id"], r["captured_at"]))
    return rows


def _enumerate_parquet_requests(parquet_path: Path) -> list[dict]:
    """Same row shape as ``_enumerate_workspace_requests`` but sourced
    from a single ``-captures`` parquet (``agentcap export`` output).
    Newer parquets carry ``task_id`` / ``turn`` so the diff / prev_rid
    chain groups per (run, task); older ones without those columns
    fall back to one linear chain per ``run_id`` and the LOC cell
    just stays ``-``."""
    import json as _json
    import pyarrow.parquet as pq

    table_meta = pq.ParquetFile(str(parquet_path)).schema_arrow
    available = set(table_meta.names)
    cols = ["request_id", "captured_at", "request", "response", "run_id"]
    has_task = "task_id" in available
    has_turn = "turn" in available
    if has_task:
        cols.append("task_id")
    if has_turn:
        cols.append("turn")
    t = pq.read_table(str(parquet_path), columns=cols)
    n = t.num_rows
    if n == 0:
        return []
    rids = t.column("request_id").to_pylist()
    times = t.column("captured_at").to_pylist()
    reqs = t.column("request").to_pylist()
    resps = t.column("response").to_pylist()
    runs = t.column("run_id").to_pylist()
    task_ids = t.column("task_id").to_pylist() if has_task else [None] * n
    turns = t.column("turn").to_pylist() if has_turn else [None] * n

    order = sorted(range(n), key=lambda i: (runs[i] or "", int(times[i] or 0)))
    rows: list[dict] = []
    prev_msgs: dict = {}
    prev_rid: dict = {}
    idx_by_key: dict = {}
    # Drop rows whose request_id isn't the proxy's 32-hex format.
    # The picker would reject them anyway (``_pick_parquet_request``
    # validates via the same regex) and they get interpolated into
    # the fzf preview shell command via ``{2}`` / ``{3}`` — keeping
    # them out at enumeration time also closes the door on any
    # injection vector from a malformed parquet.
    import re
    _hex_rid = re.compile(r"[0-9a-f]{32}")
    for i in order:
        rid = rids[i]
        if not rid or not _hex_rid.fullmatch(rid):
            continue
        try:
            body = _json.loads(reqs[i] or "{}")
        except _json.JSONDecodeError:
            body = {}
        messages = body.get("messages") or []
        run_id = runs[i] or "?"
        task_id = task_ids[i]
        # Mirror workspace semantics: group prev/diff by (run, task);
        # fall back to (run, rid) when task_id is missing so unrelated
        # rows don't chain into one synthetic task.
        key = (run_id, task_id if task_id is not None else rid)
        prior = prev_msgs.get(key)
        if prior is None:
            new_msgs = messages
            label = f"(init {len(new_msgs)})"
        else:
            removed, new_msgs = _diff_messages(prior, messages)
            label = f"({_delta_label(len(removed), len(new_msgs))})"
        summary = _message_summary(new_msgs[-1]) if new_msgs else ""
        preview = f"{label} {summary}".replace("\n", " ").strip()
        searchable = " ".join(
            _message_text(m) for m in new_msgs
        ).replace("\n", " ").replace("\t", " ")
        status = "?"
        try:
            status = str(_json.loads(resps[i] or "{}").get("status_code", "?"))
        except _json.JSONDecodeError:
            pass
        idx_by_key[key] = idx_by_key.get(key, 0) + 1
        rows.append({
            "run_id": run_id,
            "rid": rid,
            "captured_at": int(times[i] or 0),
            "status": status,
            "task_id": task_id,
            "turn": turns[i],
            "req_index": idx_by_key[key],
            "prev_rid": prev_rid.get(key),
            "preview": preview,
            "searchable": searchable,
        })
        prev_msgs[key] = messages
        prev_rid[key] = rid
    return rows


def _format_inspect_rows(
    rows: list[dict], *, include_run: bool
) -> tuple[str, list[str]]:
    """Flat table: every row is one captured call, every visible column
    is something the user might fuzzy-filter on (LOC =
    ``task_id.<req_index>``, RID, optional RUN, MESSAGES — the latter
    a ``(+N)`` / ``(init N)`` / ``(-X +Y)`` delta plus a one-line
    role-aware summary). Time / status / model / size live in the
    fzf preview pane (see ``_preview_cmd``).

    Returns ``(header, fzf_lines)``. Each fzf line is the visible
    content followed by tab-delimited hidden columns the preview
    command pulls via ``{2}`` / ``{3}`` (full rid, previous rid) plus
    a searchable blob fzf matches against (column 4)."""

    rid_w = 8
    loc_w = max(
        len("LOC"),
        max((
            len(f"{r.get('task_id') or '?'}.{r.get('req_index')}")
            if r.get("task_id") and r.get("req_index") is not None else 1
            for r in rows
        ), default=0),
    )
    run_w = (
        max(len("RUN"), max((len(r["run_id"]) for r in rows), default=0))
        if include_run else 0
    )

    def _row(loc, rid, run, prompt) -> str:
        cells = [f"{loc:<{loc_w}}", f"{rid:<{rid_w}}"]
        if include_run:
            cells.append(f"{run:<{run_w}}")
        cells.append(prompt)
        return "  ".join(cells)

    header = _row("LOC", "RID", "RUN", "MESSAGES")

    fzf: list[str] = []
    prev_task: str | None = None
    for r in rows:
        loc = (
            f"{r.get('task_id') or '?'}.{r.get('req_index')}"
            if r.get("task_id") and r.get("req_index") is not None
            else "-"
        )
        # Strip tabs from the visible content so they don't shift the
        # tab-delimited hidden columns appended below.
        line = _row(loc, r["rid"][:8], r["run_id"], r["preview"]).replace("\t", " ")
        task_id = r.get("task_id")
        if task_id and task_id != prev_task:
            # Reverse video: inverts fg/bg so the row pops on any
            # terminal palette regardless of theme.
            line = f"\033[7m{line}\033[0m"
        prev_task = task_id
        # Hidden tab columns (fzf searches all of them by default):
        #   2 = full rid, 3 = prev rid, 4 = concatenated new-message
        # bodies so a query like ``hf-cli`` matches rows whose deeper
        # content references it.
        fzf.append(
            f"{line}\t{r['rid']}\t{r.get('prev_rid') or '-'}"
            f"\t{r.get('searchable') or ''}"
        )
    return header, fzf


def _fzf_pick(
    header: str,
    lines: list[str],
    preview_cmd: str,
    *,
    extra_args: Sequence[str] = (),
) -> str | None:
    """Run fzf over ``lines`` with ``header`` pinned at the top.
    Returns the selected line, or ``None`` if the user cancelled
    (Esc / Ctrl-C). Raises ``click.UsageError`` if fzf is not on PATH
    — the gate lives here so every caller (``inspect``, ``replay``
    without a request-id, future ones) is protected automatically."""
    import shutil
    import subprocess

    if not shutil.which("fzf"):
        raise click.UsageError(
            "fzf is required for interactive pickers "
            "(install via 'brew install fzf' or your distro's package manager)."
        )

    args = [
        "fzf",
        "--ansi",
        "--layout=reverse",
        "--header", header,
        "--header-first",
        "--preview", preview_cmd,
        "--preview-window=right:60%:wrap",
        "--no-sort",
    ]
    args.extend(extra_args)
    proc = subprocess.run(
        args,
        input="\n".join(lines),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout.rstrip("\n") or None


def _pick_workspace_run() -> str | None:
    """Open an fzf picker over the runs in the workspace, returning the
    selected run-id, or ``None`` if cancelled. fzf is a hard
    requirement of ``inspect``; the gate lives at the top of
    ``inspect_cmd``."""
    import json as _json
    import sys

    root = _workspace_root()
    if not root.is_dir():
        raise click.UsageError(f"no workspace at {root}")

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
        if n_caps == 0:
            continue  # skip empty runs — nothing to inspect
        tasks = meta.get("tasks") or []
        rows.append({
            "run_id": run_dir.name,
            "agent": meta.get("agent") or "?",
            "model": (meta.get("model") or "?").split("/")[-1],
            "n_tasks": len(tasks),
            "n_caps": n_caps,
        })
    if not rows:
        raise click.UsageError(f"no runs with captures in {root}")

    run_w = max(len("RUN_ID"), max(len(r["run_id"]) for r in rows))
    agent_w = max(len("AGENT"), max(len(r["agent"]) for r in rows))
    model_w = max(len("MODEL"), max(len(r["model"]) for r in rows))

    def _row(rid, agent, model, tasks, caps) -> str:
        return (
            f"{rid:<{run_w}}  {agent:<{agent_w}}  {model:<{model_w}}  "
            f"{tasks:>5}  {caps:>4}"
        )

    header = _row("RUN_ID", "AGENT", "MODEL", "TASKS", "CAPS")
    lines = [
        _row(r["run_id"], r["agent"], r["model"], str(r["n_tasks"]), str(r["n_caps"]))
        for r in rows
    ]
    # Preview shows the run.json metadata so the user sees what's inside
    # before drilling in.
    preview = (
        f"{sys.executable} -m agentcap _run_preview {{1}} 2>/dev/null | head -200"
    )

    picked = _fzf_pick(header, lines, preview)
    if picked is None:
        return None
    tokens = picked.split()
    return tokens[0] if tokens else None


def _pick_workspace_request(
    scope: str | None, *, initial_short_rid: str | None = None,
) -> str | None:
    """fzf picker for a workspace request. Returns the picked short
    rid, or ``None`` if cancelled. fzf is a hard requirement of
    ``inspect``; the gate lives at the top of ``inspect_cmd``.

    ``initial_short_rid`` (if given) positions the cursor on the row
    whose rid starts with that prefix when the picker opens — used
    when re-entering the picker from the message sub-picker so the
    user lands back where they were."""
    import sys

    rows = _enumerate_workspace_requests(scope)
    if not rows:
        where = f"run {scope!r}" if scope else "workspace"
        raise click.UsageError(f"no captured requests in {where}")

    include_run = scope is None
    header, fzf_lines = _format_inspect_rows(rows, include_run=include_run)
    # Tab-delim hidden columns: 2 = full rid, 3 = previous-capture rid
    # (or "-" for the first capture of a task). Pre-computing the prev
    # rid here lets the preview pane skip a full cap-dir rescan per
    # fzf hover. ``_highlight`` wraps each occurrence of fzf's current
    # query (``{q}``) in red so the user can see where the match
    # landed inside the preview. ``{q}`` is its own positional arg so
    # fzf's automatic shell-escaping handles quoting end-to-end.
    preview = (
        f"{sys.executable} -m agentcap _preview {{2}} {{3}} 2>/dev/null"
        f" | head -400"
        f" | {sys.executable} -m agentcap _highlight {{q}}"
    )

    extra = [
        "--delimiter", "\t", "--with-nth", "1",
        "--no-hscroll",
        "--bind", "change:refresh-preview",
    ]
    if initial_short_rid:
        for i, line in enumerate(fzf_lines, start=1):
            parts = line.split("\t")
            # Hidden column 2 carries the full rid; match by prefix.
            if len(parts) >= 2 and parts[1].startswith(initial_short_rid):
                # ``load`` fires after fzf finishes reading stdin so
                # the items exist when ``pos(N)`` runs (``start`` is
                # too early — fires before items are loaded).
                extra.extend(["--bind", f"load:pos({i})"])
                break

    picked = _fzf_pick(
        header, fzf_lines, preview,
        extra_args=extra,
    )
    if picked is None:
        return None  # cancelled
    # picked is the visible (column-1) line; RID is the second
    # whitespace-separated field on it.
    tokens = picked.split()
    short = tokens[1] if len(tokens) >= 2 else ""
    import re
    if not re.fullmatch(r"[0-9a-f]{8}", short):
        return None
    return short


def _classify_source(source: str | None) -> tuple[str, str | None]:
    """Map ``--source`` into ``(kind, normalised_payload)``:

    - ``("workspace", None)`` — no source given.
    - ``("parquet", abs_path)`` — local ``.parquet`` file.
    - ``("hf", "<owner>/<name>")`` — ``hf://datasets/<owner>/<name>``
      or the bare ``<owner>/<name>`` shorthand (matches the syntax
      ``replay.load_request`` already accepts).

    Anything else raises ``UsageError``."""
    if source is None:
        return "workspace", None
    if source.endswith(".parquet"):
        p = Path(source)
        if not p.is_file():
            raise click.UsageError(f"parquet not found: {source}")
        return "parquet", str(p)
    s = source
    if s.startswith("hf://datasets/"):
        s = s[len("hf://datasets/"):]
    s = s.strip("/")
    # ``owner/name`` shorthand — exactly one ``/``, both parts non-empty.
    if s.count("/") == 1 and all(s.split("/")):
        return "hf", s
    raise click.UsageError(
        f"source must be a .parquet file or an hf://datasets/<owner>/<name> "
        f"URI (or the bare <owner>/<name> shorthand); got {source!r}"
    )


def _fetch_hf_parquet_meta(repo_id: str, path: str) -> dict:
    """Pull one parquet's preview-relevant metadata. Returns
    ``{model, num_rows, tasks: [{id, turns, prompt}]}``. ``tasks`` is
    empty when the parquet predates the ``task_id`` schema.

    Tries the local ``huggingface_hub`` cache first via
    ``try_to_load_from_cache`` — if the user has previously
    ``hf_hub_download``-ed this parquet (e.g. by picking it through
    inspect once already), we read the local file directly, which is
    much faster than the ``HfFileSystem`` range-reads we'd otherwise
    do over the network. On cache miss we fall back to HfFileSystem
    (footer + row-group-0 reads, no full download)."""
    import json as _json
    from huggingface_hub import HfFileSystem, try_to_load_from_cache
    import pyarrow.parquet as pq
    out: dict = {"model": None, "num_rows": 0, "tasks": []}

    local = try_to_load_from_cache(
        repo_id=repo_id, filename=path, repo_type="dataset",
    )
    # ``try_to_load_from_cache`` returns ``str``, ``_CACHED_NO_EXIST``,
    # or ``None``. Only the first is usable.
    if isinstance(local, str) and Path(local).is_file():
        opener = open(local, "rb")
    else:
        opener = HfFileSystem().open(f"datasets/{repo_id}/{path}", "rb")

    with opener as fh:
        pf = pq.ParquetFile(fh)
        out["num_rows"] = pf.metadata.num_rows
        cols = pf.schema_arrow.names
        if "model" in cols and pf.num_row_groups:
            # First row-group + single column → tiny download.
            rg = pf.read_row_group(0, columns=["model"])
            if rg.num_rows:
                out["model"] = rg.column("model")[0].as_py()
        if "task_id" in cols and pf.num_row_groups:
            # Sample tasks + prompts from row group 0 ONLY. Reading
            # ``task_id`` + ``turn`` across all ~88 row groups (one
            # per export batch) was the dominant cost in the cold
            # picker open — ~10 s per parquet × 25 parquets — for no
            # extra signal in a preview that already truncates the
            # task list at ``_HF_PREVIEW_TASK_LIMIT``. ``request`` is
            # a huge per-row column so we keep it confined to row
            # group 0 too. Tasks whose rows fall in later row groups
            # simply don't show up in the preview.
            rg_cols = ["task_id"]
            if "turn" in cols:
                rg_cols.append("turn")
            if "request" in cols:
                rg_cols.append("request")
            rg = pf.read_row_group(0, columns=rg_cols)
            tids = rg.column("task_id").to_pylist()
            turns = (
                rg.column("turn").to_pylist()
                if "turn" in rg_cols else [None] * len(tids)
            )
            raws = (
                rg.column("request").to_pylist()
                if "request" in rg_cols else [None] * len(tids)
            )
            per_task: dict[str, dict] = {}
            for tid, t, raw in zip(tids, turns, raws):
                if not tid:
                    continue
                d = per_task.setdefault(tid, {"turns": 0, "prompt": None})
                if t is not None and int(t) > d["turns"]:
                    d["turns"] = int(t)
                if d["prompt"] is None and raw:
                    try:
                        msgs = (_json.loads(raw) or {}).get("messages") or []
                    except (_json.JSONDecodeError, ValueError, TypeError):
                        msgs = []
                    for m in msgs:
                        if m.get("role") == "user":
                            d["prompt"] = _message_text(m).replace("\n", " ")
                            break
            out["tasks"] = [
                {"id": tid, "turns": per_task[tid]["turns"],
                 "prompt": per_task[tid]["prompt"]}
                for tid in sorted(per_task)
            ]
    return out


def _hf_list_parquets(repo_id: str) -> list[dict]:
    """List ``.parquet`` files in ``<owner>/<name>`` via a single
    ``HfApi().list_repo_tree`` call. Each row carries what the tree
    provides plus what we can parse out of the export filename:
    ``{path, size, agent, model, ts}``.

    Export writes ``train-<agent>-<model>-<ts>-<hash>.parquet``, so
    everything between the agent and the ts tokens is the model —
    free signal we can show in the picker without fetching the
    parquet footer.

    ``num_rows`` and the task list still require the per-parquet
    footer read and are hydrated asynchronously by
    ``_pick_hf_dataset_parquet`` into the preview pane."""
    from huggingface_hub import HfApi
    api = HfApi()
    tree = api.list_repo_tree(repo_id, repo_type="dataset", recursive=True)
    base: list[dict] = []
    for entry in tree:
        path = getattr(entry, "path", None) or getattr(entry, "rfilename", None)
        if not path or not path.endswith(".parquet"):
            continue
        size = getattr(entry, "size", None) or 0
        stem = Path(path).stem
        agent = model = ts = None
        parts = stem.split("-")
        if len(parts) >= 5 and parts[0] == "train":
            agent = parts[1]
            model = "-".join(parts[2:-2])
            ts = parts[-2]
        base.append({
            "path": path, "size": int(size),
            "agent": agent, "model": model, "ts": ts,
        })
    base.sort(key=lambda r: (r["agent"] or "", r["ts"] or "", r["path"]))
    return base


def _hf_meta_tempfile(tempdir: Path, path: str) -> Path:
    """Per-parquet path under the picker's session tempdir. The
    prefetch subprocess writes to it; the preview cmd reads from
    it. Stable filename so both sides agree without coordination.

    Uses a SHA-1 digest of the full path so distinct HF paths can't
    map to the same tempfile (``a/b__c.parquet`` and ``a__b/c.parquet``
    would otherwise collide under naive ``/`` → ``__`` replacement).
    The stem is preserved as a readable prefix for debugging."""
    import hashlib
    digest = hashlib.sha1(path.encode()).hexdigest()[:16]
    return tempdir / f"{Path(path).stem}-{digest}.json"


def _write_meta_atomic(target: Path, meta: dict) -> None:
    """Write the metadata JSON in two steps so concurrent readers
    (the fzf preview cmd) never see a half-written file."""
    import json as _json
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(_json.dumps(meta, ensure_ascii=False))
    tmp.replace(target)


def _fetch_hf_meta_into_tempdir(
    repo_id: str, path: str, tempdir: Path,
) -> None:
    """Hydrate one parquet's metadata and stash it under ``tempdir``.
    The session tempdir is per-invocation, so every call pays the
    network round-trip; there is no cross-session reuse.

    On fetch failure (HF timeout / SSL error / unreachable) we do NOT
    write a placeholder. Otherwise the preview cmd reads the empty
    placeholder and renders it as a normal-but-empty parquet, hiding
    the real cause. Leaving the file absent keeps the preview showing
    "loading…" so the user knows something's still pending."""
    target = _hf_meta_tempfile(tempdir, path)
    if target.is_file():
        return
    try:
        meta = _fetch_hf_parquet_meta(repo_id, path)
    except Exception:  # noqa: BLE001
        return
    try:
        _write_meta_atomic(target, meta)
    except OSError:
        pass  # tempdir was removed (picker exited) — silent


_HF_PREVIEW_TASK_LIMIT = 15


@cli.command("_hf_parquet_preview", hidden=True)
@click.option(
    "--tempdir", "tempdir_str", required=True,
    type=click.Path(file_okay=False, dir_okay=True),
    help="Session tempdir populated by the prefetch subprocess.",
)
@click.argument("path")
def _hf_parquet_preview_cmd(tempdir_str: str, path: str) -> None:
    """Internal: preview pane for the HF dataset parquet picker.

    Reads the cached metadata for ``path`` from a per-session
    ``--tempdir`` populated by the prefetch subprocess that
    ``_pick_hf_dataset_parquet`` launches. fzf re-spawns this command
    on every hover, so each invocation is one ``Path.read_text``.

    The command exits immediately if the metadata isn't on disk yet
    — a long-lived preview process interferes with fzf's input
    handling on some terminals. The picker's ``--listen`` port lets
    the prefetch subprocess POST ``refresh-preview`` once each file
    lands, so this short-lived render gets re-invoked automatically
    when the data is ready."""
    import json as _json
    tempdir = Path(tempdir_str)
    target = _hf_meta_tempfile(tempdir, path)

    click.echo(f"path:   {path}")
    if not target.is_file():
        click.echo("loading…")
        return

    try:
        meta = _json.loads(target.read_text())
    except (OSError, _json.JSONDecodeError) as exc:
        click.echo(f"(preview failed: {type(exc).__name__}: {exc})")
        return
    click.echo(f"rows:   {meta.get('num_rows', 0):,}")
    click.echo(f"model:  {meta.get('model') or '?'}")
    tasks = meta.get("tasks") or []
    click.echo(f"tasks:  {len(tasks)}")
    if not tasks:
        click.echo()
        click.echo("(no task_id column — pre-schema-upgrade parquet)")
        return
    click.echo()
    click.echo("─── TASKS ───")
    shown = tasks[:_HF_PREVIEW_TASK_LIMIT]
    for t in shown:
        prompt = _flatten(t.get("prompt") or "(no user message)", 120)
        click.echo(f"  {t['id']}: ({t.get('turns', 0)} turns) {prompt}")
    hidden = len(tasks) - len(shown)
    if hidden > 0:
        click.echo(f"  … and {hidden} more")


@cli.command("_hf_prefetch", hidden=True)
@click.option(
    "--tempdir", "tempdir_str", required=True,
    type=click.Path(file_okay=False, dir_okay=True),
)
@click.option("--repo", "repo_id", required=True)
@click.option(
    "--fzf-port", "fzf_port", type=int, default=None,
    help="HTTP port of fzf's --listen server, for refresh-preview "
         "after each successful fetch.",
)
def _hf_prefetch_cmd(
    tempdir_str: str, repo_id: str, fzf_port: int | None,
) -> None:
    """Internal: prefetch parquet metadata for the picker into the
    session tempdir.

    Run as a subprocess by ``_pick_hf_dataset_parquet`` so the picker
    can SIGKILL it when fzf exits (Esc / pick) instead of waiting on
    in-flight network IO. Reads a JSON array of parquet paths from
    stdin, hydrates each serially in list order, and writes the
    results atomically to ``<tempdir>/<safe-path>.json`` — exactly
    where ``_hf_parquet_preview_cmd`` looks.

    After each successful write we POST ``refresh-preview`` to fzf's
    ``--listen`` port. fzf then re-runs the preview cmd for whatever
    row the user has focused; if that row's metadata just landed,
    the preview pane updates without any keyboard input. POSTs that
    can't be delivered (fzf already exited, race during startup,
    user's network stack is unhappy) are swallowed silently — the
    fetch already succeeded; the missed refresh is recoverable by
    bumping the cursor.

    The parent silences stderr (``subprocess.DEVNULL``) so retry
    warnings emitted by ``huggingface_hub`` for transient errors and
    socket-close races during shutdown never reach the user's
    terminal."""
    import json as _json
    import sys as _sys
    import urllib.error
    import urllib.request
    tempdir = Path(tempdir_str)
    try:
        paths = _json.loads(_sys.stdin.read())
    except (OSError, _json.JSONDecodeError):
        return

    def _nudge_fzf() -> None:
        if fzf_port is None:
            return
        req = urllib.request.Request(
            f"http://127.0.0.1:{fzf_port}/",
            data=b"refresh-preview",
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=0.5) as resp:
                resp.read()
        except (urllib.error.URLError, OSError):
            pass  # fzf not up yet, or already exited — harmless

    # Serial fetch: strict list-order completion. The previous 8-way
    # parallel path was provably hitting HF endpoint-side timeouts
    # (TLS handshake / read timeout) for a subset of parquets on
    # every run, with each retry adding 1-15 s. Specific parquets
    # reproducibly hit those retries; serial access avoids the
    # contention entirely.
    for path in paths:
        target = _hf_meta_tempfile(tempdir, path)
        already = target.is_file()
        _fetch_hf_meta_into_tempdir(repo_id, path, tempdir)
        # Only nudge fzf on a fresh successful write — otherwise we
        # spam refresh-preview for files we didn't touch.
        if not already and target.is_file():
            _nudge_fzf()


def _pick_hf_dataset_parquet(repo_id: str, tempdir: Path) -> Path | None:
    """Level-1 picker over the ``.parquet`` files in an HF dataset.
    Returns the local cached path of the picked parquet, or ``None``
    if cancelled.

    ``tempdir`` is owned by the CALLER (``inspect_cmd``) and persists
    across re-entries to this picker — pressing Esc on the
    request-level picker re-opens this one with all the previously
    fetched previews still in place. A fresh subprocess is launched
    on each entry, but it fast-skips already-fetched paths via the
    early-return in ``_fetch_hf_meta_into_tempdir``.

    Loading strategy: ``list_repo_tree`` returns the file list cheaply
    (~1 s); fzf opens with that immediately. The expensive per-parquet
    footer reads (~1 s each over HfFileSystem, serial to avoid the
    parallel-overload-driven HF retries we measured at workers=8) run
    in a single ``_hf_prefetch`` subprocess. After each successful
    write the subprocess POSTs ``refresh-preview`` to fzf's
    ``--listen`` port, so the preview pane auto-updates the instant
    the focused row's metadata lands — no cursor bump required.

    When fzf exits (pick or Esc) we SIGKILL the subprocess. Nothing
    under ``tempdir`` is shared state; ``stderr=DEVNULL`` keeps the
    HF retry chatter off the user's terminal."""
    import json as _json
    import shlex
    import socket
    import subprocess
    import sys
    from huggingface_hub import hf_hub_download

    rows = _hf_list_parquets(repo_id)
    if not rows:
        raise click.UsageError(f"no .parquet files in {repo_id}")

    def _size_label(n: int) -> str:
        for unit in ("B", "K", "M", "G"):
            if n < 1024 or unit == "G":
                return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
            n /= 1024
        return f"{n}"

    # Display columns ``AGENT  SIZE  MODEL`` — path is in the hidden
    # tab-column for the preview lookup, but isn't useful for visual
    # selection (it shows up in the preview pane anyway). ``ROWS``
    # and the task list still need the footer fetch and live in the
    # preview pane.
    agent_w = max(len("AGENT"), max(len(r["agent"] or "?") for r in rows))
    model_w = max(len("MODEL"), max(len(r["model"] or "?") for r in rows))

    def _fmt(agent, size, model) -> str:
        return f"{agent:<{agent_w}}  {size:>7}  {model:<{model_w}}"

    header = _fmt("AGENT", "SIZE", "MODEL")
    lines = [
        _fmt(r["agent"] or "?", _size_label(r["size"]), r["model"] or "?")
        + f"\t{r['path']}"
        for r in rows
    ]

    # Pre-allocate a port for fzf's --listen HTTP API. We pass it
    # to both fzf (so fzf listens on it) and to the prefetch
    # subprocess (so it can POST ``refresh-preview`` after each
    # file lands). There's a brief race between closing this
    # socket and fzf binding, but it's local-only and the
    # subprocess silently ignores connection errors anyway.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        fzf_port = s.getsockname()[1]

    # Background prefetch for ALL rows via ``_hf_prefetch``. The
    # picker opens after just the ``list_repo_tree`` call (~1 s) and
    # previews fill in top-down as the subprocess writes them. Its
    # stderr is /dev/null so urllib3/HfHub retry warnings don't
    # surface. Each successful write triggers an HTTP POST to fzf's
    # listen port; fzf re-runs the preview cmd for the focused row
    # and the pane auto-updates without the user touching the keyboard.
    manifest = _json.dumps([r["path"] for r in rows])
    proc = subprocess.Popen(
        [sys.executable, "-m", "agentcap", "_hf_prefetch",
         "--tempdir", str(tempdir), "--repo", repo_id,
         "--fzf-port", str(fzf_port)],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        assert proc.stdin is not None
        proc.stdin.write(manifest.encode())
        proc.stdin.close()
    except (BrokenPipeError, OSError):
        pass  # subprocess died early — we'll detect it via poll()

    preview = (
        f"{sys.executable} -m agentcap _hf_parquet_preview"
        f" --tempdir={shlex.quote(str(tempdir))} {{2}} 2>/dev/null"
    )
    try:
        picked = _fzf_pick(
            header, lines, preview,
            extra_args=[
                "--delimiter", "\t", "--with-nth", "1",
                "--no-hscroll",
                f"--listen=127.0.0.1:{fzf_port}",
            ],
        )
    finally:
        # SIGKILL is decisive: takes the subprocess down instantly.
        # We then ``wait`` briefly to reap the zombie. ``tempdir``
        # outlives this call (the caller manages it) so any partial
        # writes the subprocess made are kept for the next entry.
        if proc.poll() is None:
            proc.kill()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
    if picked is None:
        return None
    rel = picked.rsplit("\t", 1)[-1].strip()

    # ``hf_hub_download`` is a no-op when the file is already cached
    # under ~/.cache/huggingface — no extra plumbing needed for the
    # "cache between picker invocations" requirement.
    return Path(hf_hub_download(
        repo_id=repo_id, repo_type="dataset", filename=rel,
    ))


def _pick_parquet_request(parquet_path: Path) -> str | None:
    """fzf picker over the rows of a captures parquet. Same shape as
    ``_pick_workspace_request`` but the preview pipeline shells out to
    ``_preview_parquet`` (which reads from the parquet) instead of
    ``_preview`` (which scans the workspace). Returns the picked
    FULL rid or ``None`` if cancelled. Unlike the workspace flow
    (which accepts an 8-char prefix because ``resolve_workspace_rid``
    expands it), the parquet-source path through ``_resolve_request_id``
    does an exact-match lookup, so we must return the full rid."""
    import shlex
    import sys

    rows = _enumerate_parquet_requests(parquet_path)
    if not rows:
        raise click.UsageError(f"no rows in {parquet_path}")
    header, fzf_lines = _format_inspect_rows(rows, include_run=True)
    pq_quoted = shlex.quote(str(parquet_path))
    preview = (
        f"{sys.executable} -m agentcap _preview_parquet {pq_quoted}"
        f" {{2}} {{3}} 2>/dev/null"
        f" | head -400"
        f" | {sys.executable} -m agentcap _highlight {{q}}"
    )
    extra = [
        "--delimiter", "\t", "--with-nth", "1",
        "--no-hscroll",
        "--bind", "change:refresh-preview",
    ]
    picked = _fzf_pick(header, fzf_lines, preview, extra_args=extra)
    if picked is None:
        return None
    # Hidden tab-delim column 2 carries the full 32-char rid
    # (set by ``_format_inspect_rows``). Avoid the visible 8-char
    # prefix — the parquet's request_id column stores full rids.
    fields = picked.split("\t")
    import re
    full_rid = fields[1] if len(fields) >= 2 else ""
    if not re.fullmatch(r"[0-9a-f]{32}", full_rid):
        return None
    return full_rid


@cli.command("inspect")
@click.argument("target", required=False, shell_complete=_complete_request_ids)
@click.option(
    "--source",
    default=None,
    help="Where to look up the request: a capture dir, a .parquet, or "
    "hf://datasets/<owner>/<name>. Only honored when TARGET is a "
    "request-id; defaults to scanning the local workspace.",
)
@click.option(
    "--rid",
    "print_rid_only",
    is_flag=True,
    help="When picking interactively, print only the selected request-id "
    "(useful for piping into `agentcap replay`).",
)
def inspect_cmd(target: str | None, source: str | None, print_rid_only: bool) -> None:
    """Inspect captured requests.

    \b
    - ``agentcap inspect``              pick a run, then a call (two-step)
    - ``agentcap inspect <run-id>``     pick a call from that run
    - ``agentcap inspect <rid>``        print the captured body for that request

    A rid is 32 hex chars (proxy-minted UUID); a run-id contains a dash.
    The interactive pickers require fzf on PATH (``_fzf_pick`` errors
    out with a clear message if it isn't).
    """
    import json as _json

    # rid (full or short prefix) → body dump (single request).
    if target and "-" not in target:
        full_rid, body, resp_rec, _, _ = _resolve_request_id(target, source)
        if resp_rec is not None:
            click.echo(
                f"  request_id={full_rid} "
                f"captured_at={resp_rec.get('captured_at_resp', '?')} "
                f"status={resp_rec.get('status_code', '?')}",
                err=True,
            )
        click.echo(_json.dumps(body, indent=2, ensure_ascii=False))
        return

    # ``--source`` (no target): drive a parquet- or hf-dataset-rooted
    # picker chain instead of the workspace. Esc walks back one level
    # at a time, same as the workspace flow:
    #   hf-dataset:  message → request → parquet → exit
    #   parquet:     message → request → exit
    kind, payload = _classify_source(source) if target is None else ("workspace", None)
    if kind in ("parquet", "hf"):
        # For ``hf`` we hold a tempdir at THIS scope so each Esc-back
        # from the request picker re-enters the parquet picker with
        # all previously fetched previews still on disk. ``parquet``
        # case doesn't need it — its source IS the parquet path.
        import contextlib as _contextlib
        import tempfile as _tempfile
        if kind == "hf":
            td_cm = _tempfile.TemporaryDirectory(prefix="agentcap-hf-meta-")
        else:
            td_cm = _contextlib.nullcontext(None)
        with td_cm as td_str:
            hf_tempdir = Path(td_str) if td_str else None
            while True:
                if kind == "hf":
                    pq_path = _pick_hf_dataset_parquet(payload, hf_tempdir)  # type: ignore[arg-type]
                    if pq_path is None:
                        return  # Esc on the parquet picker → exit
                else:
                    pq_path = Path(payload)  # type: ignore[arg-type]
                pq_source = str(pq_path)
                while True:
                    pick = _pick_parquet_request(pq_path)
                    if pick is None:
                        break  # Esc on the request picker → back one level
                    if print_rid_only:
                        full_rid, _, _, _, _ = _resolve_request_id(pick, pq_source)
                        click.echo(full_rid)
                        return
                    _pick_request_message(pick, source=pq_source)
                if kind == "parquet":
                    return  # explicit parquet on CLI; Esc → exit

    # Esc walks back one level: message picker → request picker → run
    # picker → exit. There's no single-key skip; pressing Esc three
    # times from the deepest level exits. A run-id passed on the CLI
    # pins the request loop to that run (Esc on the request picker
    # exits instead of falling back to a run picker).
    cli_target = target

    while True:
        scope = cli_target
        if scope is None:
            scope = _pick_workspace_run()
            if scope is None:
                return  # Esc on the run picker → exit
        # Drill into the selected run: pick a request, then drill into
        # its flattened conversation. Esc on the message sub-picker
        # returns here; the request picker re-opens with the cursor on
        # the same row the user just visited.
        last_pick: str | None = None
        while True:
            pick = _pick_workspace_request(
                scope, initial_short_rid=last_pick,
            )
            if pick is None:
                break  # Esc on the request picker → back one level
            last_pick = pick
            if print_rid_only:
                full_rid, _, _, _, _ = _resolve_request_id(pick, None)
                click.echo(full_rid)
                return
            _pick_request_message(pick)
        if cli_target is not None:
            return  # explicit run-id on CLI; Esc → exit


@cli.command("_run_preview", hidden=True)
@click.argument("run_id")
def _run_preview_cmd(run_id: str) -> None:
    """Internal: preview a run's metadata for the run picker."""
    import json as _json

    run_dir = _workspace_root() / run_id
    meta_path = run_dir / "run.json"
    if not meta_path.is_file():
        click.echo(f"(no run.json at {meta_path})")
        return
    try:
        meta = _json.loads(meta_path.read_text())
    except (OSError, _json.JSONDecodeError) as exc:
        click.echo(f"(run.json unreadable: {exc})")
        return
    captures = run_dir / "captures"
    n_caps = (
        len(list(captures.glob("*.request.json"))) if captures.is_dir() else 0
    )
    click.echo(f"run:       {run_id}")
    click.echo(f"agent:     {meta.get('agent', '?')}")
    click.echo(f"model:     {meta.get('model', '?')}")
    click.echo(f"upstream:  {meta.get('upstream', '?')}")
    click.echo(f"followup:  {meta.get('followup', '?')}")
    click.echo(f"turns/task: {meta.get('turns_per_task', '?')}")
    click.echo(f"captures:  {n_caps}")
    click.echo()
    click.echo("─── TASKS ───")
    for t in meta.get("tasks") or []:
        prompt = (t.get("prompt") or "").replace("\n", " ")
        completed = t.get("completed_turns", "?")
        click.echo(f"  {t.get('task_id', '?')}: ({completed} turns) {prompt}")


def _message_key(m: dict) -> tuple:
    """Canonical key for a ``messages[]`` entry. Compares only the
    load-bearing fields (role/content/tool_call_id/tool_calls); ignores
    optional metadata like the tool ``name`` field that some agents
    include on one turn but not the next (notably hermes when it
    re-serialises its session DB across turn boundaries)."""
    import json as _json
    c = m.get("content")
    if isinstance(c, list):
        c = _json.dumps(c, sort_keys=True)
    tc = m.get("tool_calls")
    tc_key = _json.dumps(tc, sort_keys=True) if tc else None
    return (m.get("role"), c, m.get("tool_call_id"), tc_key)


def _diff_messages(prev: list, curr: list) -> tuple[list, list]:
    """``(removed, added)`` — the suffixes of ``prev`` and ``curr`` that
    diverge. Element-by-element so a length-equal turn boundary (where
    an agent swaps a meta-prompt for the user's followup at the last
    index) shows up as a real diff. Pure-append cases yield
    ``removed=[]``; swaps yield non-empty removed AND added of equal
    or unequal length depending on the truncation.
    """
    prev_keys = [_message_key(m) for m in prev]
    curr_keys = [_message_key(m) for m in curr]
    n = min(len(prev_keys), len(curr_keys))
    i = n
    for j in range(n):
        if prev_keys[j] != curr_keys[j]:
            i = j
            break
    return prev[i:], curr[i:]


def _delta_label(removed: int, added: int) -> str:
    """Compact ``messages[]`` delta marker. Hides the removed count
    when zero (the common pure-append case) so mid-loop rows stay
    visually quiet; surfaces it for swaps (e.g. ``-1 +1``)."""
    if removed:
        return f"-{removed} +{added}"
    return f"+{added}"


def _message_text(m: dict) -> str:
    """Flatten ``message.content`` to a string. Tool / multimodal
    messages carry list-typed content; join the text parts."""
    c = m.get("content")
    if isinstance(c, list):
        return " ".join(
            p.get("text", "") for p in c if isinstance(p, dict)
        )
    return c or ""


def _flatten(s: str, cap: int) -> str:
    """Single-line, length-capped text. Without this, content with
    embedded newlines (assistant prose, tool outputs) would blow up to
    many visible lines and push later messages off fzf's preview
    window."""
    s = " ".join(s.split())
    return s if len(s) <= cap else s[:cap] + "…"


_PICKER_SUMMARY_CAP = 160
_PREVIEW_MSG_CAP = 400


def _tag(label: str) -> str:
    """Reverse-video the ``[label]`` marker that introduces each
    preview line so the role boundaries are visually scannable across
    many similar-looking rows."""
    return f"\033[7m[{label}]\033[0m"


def _message_summary(m: dict) -> str:
    """One-line role-aware summary of one ``messages[]`` entry. Used
    in the picker's MESSAGES column where we have ~one row to convey
    'what's new in this call'. Truncated so a large tool result can't
    bloat the row."""
    role = (m or {}).get("role", "?")
    if role == "assistant":
        tcs = m.get("tool_calls") or []
        if tcs:
            tc = tcs[0]
            fn = (tc.get("function") or {}).get("name") or "?"
            args = (tc.get("function") or {}).get("arguments") or ""
            extra = f" +{len(tcs)-1}" if len(tcs) > 1 else ""
            s = f"assistant→{fn}{extra} {args}"
        else:
            s = f"assistant: {_message_text(m)}"
    elif role == "tool":
        s = f"tool: {_message_text(m)}"
    else:
        s = f"{role}: {_message_text(m)}"
    return _flatten(s, _PICKER_SUMMARY_CAP)


def _render_preview_message(m: dict) -> None:
    """Render one ``messages[]`` entry into the inspect preview pane.
    Each message stays on one line (newlines collapsed) so the diff
    suffix remains visible inside fzf's 60% pane. ``color=True`` on
    every echo: this command's stdout is captured by fzf's preview
    subprocess (not a TTY), and click strips ANSI by default in that
    case, which would silently swallow the reverse-video markers."""
    role = m.get("role", "?")
    if role == "assistant":
        for tc in m.get("tool_calls") or []:
            fn = (tc.get("function") or {}).get("name") or "?"
            args = (tc.get("function") or {}).get("arguments") or ""
            click.echo(
                f"  {_tag(f'assistant tool_call → {fn}')}  args={_flatten(args, 240)}",
                color=True,
            )
        content = _message_text(m)
        if content:
            click.echo(
                f"  {_tag('assistant content')} {_flatten(content, _PREVIEW_MSG_CAP)}",
                color=True,
            )
        return
    if role == "tool":
        tcid = (m.get("tool_call_id") or "?")[:8]
        click.echo(f"  {_tag(f'tool result, tool_call_id={tcid}')}", color=True)
        click.echo(f"  {_flatten(_message_text(m), _PREVIEW_MSG_CAP)}", color=True)
        return
    click.echo(
        f"  {_tag(role)} {_flatten(_message_text(m), _PREVIEW_MSG_CAP)}",
        color=True,
    )


@cli.command("_preview", hidden=True)
@click.argument("request_id")
@click.argument("prev_request_id", required=False, default=None)
def _preview_cmd(request_id: str, prev_request_id: str | None) -> None:
    """Internal: header + initial PROMPT + MESSAGES diff for one
    captured request — used by the fzf preview pane.

    Not part of the public CLI surface — hidden from ``--help``. The
    user-facing inspector is ``agentcap inspect <rid>``.

    ``prev_request_id`` is pushed in by the picker so the preview can
    load the diff base directly instead of scanning the capture dir on
    every fzf hover. Accepts ``"-"`` (or absent) for "no previous".
    """
    import json as _json
    import re

    # Hovered a section-header line in the picker — render nothing.
    if not re.fullmatch(r"[0-9a-f]+", request_id):
        click.echo("(section header — navigate to a request id)")
        return

    full_rid, body, resp_rec, req_rec, cap_dir = _resolve_request_id(request_id, None)
    messages = body.get("messages") or []
    initial_user = next(
        (m for m in messages if m.get("role") == "user"),
        None,
    )
    initial_prompt = _message_text(initial_user or {})
    import time as _time

    status = (
        resp_rec.get("status_code") if resp_rec is not None else "?"
    )
    serialized = _json.dumps(body, ensure_ascii=False)
    size_b = len(serialized.encode("utf-8"))
    task_id = (req_rec or {}).get("task_id")
    turn = (req_rec or {}).get("turn")
    captured_at = (req_rec or {}).get("captured_at")
    ts = (
        _time.strftime("%H:%M:%S", _time.gmtime(int(captured_at)))
        if captured_at else "?"
    )
    # Load the diff base directly from the prev-rid file in the same
    # capture dir (already known from ``_resolve_request_id`` above —
    # no second workspace scan). The picker pushes the predecessor's
    # rid in as ``prev_request_id``. Reject anything that isn't
    # lowercase hex so a hand-crafted arg can't escape the capture
    # dir via ``..`` or absolute paths.
    prev_messages: list = []
    has_previous = False
    if (
        cap_dir is not None
        and prev_request_id
        and prev_request_id != "-"
        and re.fullmatch(r"[0-9a-f]+", prev_request_id)
    ):
        prev_path = cap_dir / f"{prev_request_id}.request.json"
        if prev_path.is_file():
            try:
                prev_rec = _json.loads(prev_path.read_text())
                prev_messages = (prev_rec.get("body") or {}).get("messages") or []
                has_previous = True
            except (OSError, _json.JSONDecodeError):
                pass
    click.echo(f"rid:    {full_rid}")
    if task_id is not None or turn is not None:
        click.echo(f"task:   {task_id or '?'}  turn={turn if turn is not None else '?'}")
    click.echo(f"time:   {ts}")
    click.echo(f"status: {status}")
    click.echo(f"model:  {body.get('model', '?')}")
    click.echo(f"size:   {size_b:,} bytes (~{size_b // 4:,} tokens)")
    click.echo()
    click.echo("─── PROMPT ──────────────────────────────────────────────")
    click.echo(initial_prompt or "(no user message)")
    click.echo()
    removed_messages, new_messages = _diff_messages(prev_messages, messages)
    if has_previous:
        header_suffix = (
            f"{_delta_label(len(removed_messages), len(new_messages))} "
            f"since previous call"
        )
    else:
        n = len(new_messages)
        header_suffix = f"initial: {n} msg{'' if n == 1 else 's'}"
    click.echo(f"─── MESSAGES ({header_suffix}) ──────────")
    if has_previous:
        # Signals that the prior history (in prev_messages) was
        # elided; what follows is the diff, not the whole conversation.
        click.echo("  ...")
    if not new_messages and not removed_messages:
        click.echo("(no diff vs previous call)")
    for m in new_messages:
        _render_preview_message(m)


def _load_parquet_body(parquet_path: Path, rid: str) -> tuple[dict, dict, int | None, str | None]:
    """Pull one request out of a captures parquet. Returns
    ``(body, resp_rec, captured_at, run_id)``.

    The parquet's ``response`` column has two shapes depending on
    whether the upstream streamed:

      - stream:    ``{"stream": True, "raw": "<SSE bytes>"}``
      - non-stream: the bare OpenAI body dict (no wrapper)

    Workspace ``*.response.json`` records always have a ``body`` key,
    and ``_decode_response`` follows that convention. Normalise the
    non-stream parquet shape into ``{"stream": False, "body": ...}``
    here so callers (notably ``_decode_response`` /
    ``_request_messages_for_view``) get the model reply rendered."""
    import json as _json
    import pyarrow.parquet as pq

    t = pq.read_table(
        str(parquet_path),
        columns=["request_id", "captured_at", "request", "response", "run_id"],
        filters=[("request_id", "=", rid)],
    )
    if t.num_rows == 0:
        return {}, {}, None, None
    try:
        body = _json.loads(t.column("request")[0].as_py() or "{}")
    except _json.JSONDecodeError:
        body = {}
    try:
        raw_resp = _json.loads(t.column("response")[0].as_py() or "{}")
    except _json.JSONDecodeError:
        raw_resp = {}
    if raw_resp.get("stream"):
        resp = raw_resp
    else:
        resp = {"stream": False, "body": raw_resp}
    ts = t.column("captured_at")[0].as_py()
    run_id = t.column("run_id")[0].as_py()
    return body, resp, (int(ts) if ts is not None else None), run_id


@cli.command("_preview_parquet", hidden=True)
@click.argument("parquet_path")
@click.argument("request_id")
@click.argument("prev_request_id", required=False, default=None)
def _preview_parquet_cmd(
    parquet_path: str, request_id: str, prev_request_id: str | None,
) -> None:
    """Internal: same preview as ``_preview`` but sourced from a
    parquet file. The picker passes the parquet path as a leading arg
    so this hidden command stays stateless."""
    import json as _json
    import re
    import time as _time

    if not re.fullmatch(r"[0-9a-f]+", request_id):
        click.echo("(section header — navigate to a request id)")
        return
    pq_path = Path(parquet_path)
    body, resp, captured_at, run_id = _load_parquet_body(pq_path, request_id)
    messages = body.get("messages") or []
    initial_user = next(
        (m for m in messages if m.get("role") == "user"),
        None,
    )
    initial_prompt = _message_text(initial_user or {})
    status = resp.get("status_code", "?") if resp else "?"
    serialized = _json.dumps(body, ensure_ascii=False)
    size_b = len(serialized.encode("utf-8"))
    ts = (
        _time.strftime("%H:%M:%S", _time.gmtime(captured_at))
        if captured_at else "?"
    )

    prev_messages: list = []
    has_previous = False
    if (
        prev_request_id
        and prev_request_id != "-"
        and re.fullmatch(r"[0-9a-f]+", prev_request_id)
    ):
        prev_body, _, _, _ = _load_parquet_body(pq_path, prev_request_id)
        prev_messages = prev_body.get("messages") or []
        has_previous = bool(prev_messages)

    click.echo(f"rid:    {request_id}")
    if run_id is not None:
        click.echo(f"run:    {run_id}")
    click.echo(f"time:   {ts}")
    click.echo(f"status: {status}")
    click.echo(f"model:  {body.get('model', '?')}")
    click.echo(f"size:   {size_b:,} bytes (~{size_b // 4:,} tokens)")
    click.echo()
    click.echo("─── PROMPT ──────────────────────────────────────────────")
    click.echo(initial_prompt or "(no user message)")
    click.echo()
    removed_messages, new_messages = _diff_messages(prev_messages, messages)
    if has_previous:
        header_suffix = (
            f"{_delta_label(len(removed_messages), len(new_messages))} "
            f"since previous call"
        )
    else:
        n = len(new_messages)
        header_suffix = f"initial: {n} msg{'' if n == 1 else 's'}"
    click.echo(f"─── MESSAGES ({header_suffix}) ──────────")
    if has_previous:
        click.echo("  ...")
    if not new_messages and not removed_messages:
        click.echo("(no diff vs previous call)")
    for m in new_messages:
        _render_preview_message(m)


def _decode_sse_response(raw: str) -> dict:
    """Decode an OpenAI-compatible SSE response stream into a single
    synthesized assistant message: ``{content, tool_calls,
    finish_reason}``. Concatenates ``delta.content`` chunks; merges
    ``delta.tool_calls`` chunks by their ``index`` field (the first
    chunk for an index carries id + function.name; later chunks
    accumulate ``function.arguments`` string fragments)."""
    import json as _json
    content_parts: list[str] = []
    tool_calls_by_idx: dict[int, dict] = {}
    finish_reason: str | None = None
    for line in raw.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            obj = _json.loads(payload)
        except (_json.JSONDecodeError, ValueError):
            continue
        for ch in obj.get("choices") or []:
            delta = ch.get("delta") or {}
            if delta.get("content"):
                content_parts.append(delta["content"])
            for tc_delta in delta.get("tool_calls") or []:
                idx = tc_delta.get("index", 0)
                slot = tool_calls_by_idx.setdefault(idx, {
                    "id": "", "type": "function",
                    "function": {"name": "", "arguments": ""},
                })
                if tc_delta.get("id"):
                    slot["id"] = tc_delta["id"]
                if tc_delta.get("type"):
                    slot["type"] = tc_delta["type"]
                fn = tc_delta.get("function") or {}
                if fn.get("name"):
                    slot["function"]["name"] = fn["name"]
                if fn.get("arguments"):
                    slot["function"]["arguments"] += fn["arguments"]
            if ch.get("finish_reason"):
                finish_reason = ch["finish_reason"]
    return {
        "content": "".join(content_parts),
        "tool_calls": [tool_calls_by_idx[k] for k in sorted(tool_calls_by_idx)],
        "finish_reason": finish_reason,
    }


def _decode_response(resp_rec: dict) -> dict:
    """Synthesize an assistant message from a response record. Handles
    both non-stream (``body.choices[0].message``) and stream (raw SSE
    bytes in ``raw``)."""
    if resp_rec.get("stream"):
        return _decode_sse_response(resp_rec.get("raw") or "")
    body = resp_rec.get("body") or {}
    ch = (body.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    return {
        "content": msg.get("content") or "",
        "tool_calls": msg.get("tool_calls") or [],
        "finish_reason": ch.get("finish_reason"),
    }


def _request_messages_for_view(
    body: dict, resp_rec: dict | None
) -> list[dict]:
    """Flatten ``messages[]`` + decoded response into one record per
    picker row. Each assistant ``tool_calls`` produces its own row
    followed (if present) by a row for the assistant's content; the
    decoded model response is appended at the end as the final
    assistant turn so the viewer shows the model's reply inline.

    Each record: ``{msg_idx, role, summary, content, ...}``. ``msg_idx``
    is the index into the original ``messages[]`` (or ``None`` for the
    synthesized response rows)."""
    records: list[dict] = []
    msgs = body.get("messages") or []
    for i, m in enumerate(msgs):
        role = m.get("role", "?")
        if role == "assistant":
            for tc in m.get("tool_calls") or []:
                fn = (tc.get("function") or {}).get("name") or "?"
                args = (tc.get("function") or {}).get("arguments") or ""
                records.append({
                    "msg_idx": i,
                    "role": f"assistant→{fn}",
                    "summary": args,
                    "content": args,
                    "tool_call_id": tc.get("id"),
                })
            content = _message_text(m)
            if content:
                records.append({
                    "msg_idx": i,
                    "role": "assistant",
                    "summary": content,
                    "content": content,
                })
            continue
        if role == "tool":
            content = _message_text(m)
            records.append({
                "msg_idx": i,
                "role": "tool",
                "summary": content,
                "content": content,
                "tool_call_id": m.get("tool_call_id"),
            })
            continue
        content = _message_text(m)
        records.append({
            "msg_idx": i,
            "role": role,
            "summary": content,
            "content": content,
        })
    if resp_rec is not None:
        decoded = _decode_response(resp_rec)
        for tc in decoded.get("tool_calls") or []:
            fn = (tc.get("function") or {}).get("name") or "?"
            args = (tc.get("function") or {}).get("arguments") or ""
            records.append({
                "msg_idx": None,
                "role": f"response→{fn}",
                "summary": args,
                "content": args,
                "tool_call_id": tc.get("id"),
            })
        content = decoded.get("content") or ""
        if content:
            records.append({
                "msg_idx": None,
                "role": "response",
                "summary": content,
                "content": content,
                "finish_reason": decoded.get("finish_reason"),
            })
    return records


def _render_msg_preview(records: list[dict], row: int) -> None:
    """Echo one entry from a message list — shared between the
    workspace- and parquet-sourced ``_msg_preview*`` commands."""
    if row < 1 or row > len(records):
        click.echo(f"(row {row} out of range; have {len(records)})")
        return
    rec = records[row - 1]
    click.echo(f"role:         {rec['role']}")
    if rec.get("msg_idx") is not None:
        click.echo(f"msg_idx:      {rec['msg_idx']}")
    else:
        click.echo("msg_idx:      (response)")
    if rec.get("tool_call_id"):
        click.echo(f"tool_call_id: {rec['tool_call_id']}")
    if rec.get("finish_reason"):
        click.echo(f"finish_reason: {rec['finish_reason']}")
    click.echo()
    click.echo(rec.get("content") or "(no content)")


@cli.command("_msg_preview", hidden=True)
@click.argument("request_id")
@click.argument("row", type=int)
def _msg_preview_cmd(request_id: str, row: int) -> None:
    """Internal: render one message (1-indexed ``row``) from the
    request's flattened message list. Used by the workspace-sourced
    message sub-picker."""
    import re
    if not re.fullmatch(r"[0-9a-f]+", request_id):
        click.echo("(invalid request id)")
        return
    _, body, resp_rec, _, _ = _resolve_request_id(request_id, None)
    _render_msg_preview(_request_messages_for_view(body, resp_rec), row)


@cli.command("_msg_preview_parquet", hidden=True)
@click.argument("parquet_path")
@click.argument("request_id")
@click.argument("row", type=int)
def _msg_preview_parquet_cmd(
    parquet_path: str, request_id: str, row: int,
) -> None:
    """Internal: same as ``_msg_preview`` but sourced from a parquet.
    Parquet ``response`` column is a JSON blob (no streaming wrapper),
    so we pass it through unchanged — ``_decode_response`` handles
    both the non-stream and the SSE-wrapped shapes."""
    import re
    if not re.fullmatch(r"[0-9a-f]+", request_id):
        click.echo("(invalid request id)")
        return
    body, resp, _, _ = _load_parquet_body(Path(parquet_path), request_id)
    _render_msg_preview(_request_messages_for_view(body, resp or None), row)


def _pick_request_message(rid: str, *, source: str | None = None) -> None:
    """Second-level fzf picker over the messages of the request the
    user selected in the request picker. Read-only browse: Esc / Enter
    both return to the caller without side effects.

    ``source`` is ``None`` for workspace-sourced rids (current
    behaviour) and a local parquet path for parquet-sourced rids — the
    preview pipeline shells out to a different hidden command in each
    case so the picker doesn't need to know how the body was loaded."""
    import shlex
    import sys
    # ``_resolve_request_id`` returns ``resp_rec=None`` for any
    # ``source`` (it calls ``replay.load_request`` which only loads the
    # request body). For parquet sources, read the response back via
    # ``_load_parquet_body`` so the message picker can show the model
    # reply rows synthesised by ``_request_messages_for_view``.
    if source and source.endswith(".parquet"):
        body, resp_rec, _, _ = _load_parquet_body(Path(source), rid)
        full_rid = rid
    else:
        full_rid, body, resp_rec, _, _ = _resolve_request_id(rid, source)
    records = _request_messages_for_view(body, resp_rec)
    if not records:
        click.echo("(no messages in this request)")
        return
    role_w = max(len(r["role"]) for r in records)
    lines: list[str] = []
    for i, rec in enumerate(records, start=1):
        summary = _flatten(rec.get("summary") or "", 200)
        display = f"[{i:>3}]  {rec['role']:<{role_w}s}  {summary}"
        # Hidden tab-delimited column 2 carries the 1-indexed row.
        # The preview reads it as ``{2}`` instead of computing
        # ``$(({n} + 1))`` — the latter is POSIX-arithmetic and fzf
        # runs previews via ``$SHELL -c``, so fish would break.
        lines.append(f"{display}\t{i}")
    header = f"messages for {full_rid[:8]} ({len(records)} entries)"
    if source and source.endswith(".parquet"):
        preview = (
            f"{sys.executable} -m agentcap _msg_preview_parquet"
            f" {shlex.quote(source)} {full_rid} {{2}} 2>/dev/null"
            f" | {sys.executable} -m agentcap _highlight {{q}}"
        )
    else:
        preview = (
            f"{sys.executable} -m agentcap _msg_preview"
            f" {full_rid} {{2}} 2>/dev/null"
            f" | {sys.executable} -m agentcap _highlight {{q}}"
        )
    _fzf_pick(
        header, lines, preview,
        extra_args=[
            "--delimiter", "\t", "--with-nth", "1",
            "--no-hscroll",
            "--bind", "change:refresh-preview",
        ],
    )


def _parse_fzf_terms(query: str) -> list[str]:
    """Split fzf's query into the literal text of each non-negated
    term. Each term has its operator prefix (``'``, ``^``) and
    trailing anchor (``$``) stripped so the remainder is the substring
    to highlight. Negated terms (``!word``) and bare ``|`` OR
    separators are skipped — they aren't substrings to colour."""
    terms: list[str] = []
    for raw in query.split():
        if raw in ("|", ""):
            continue
        if raw.startswith("!"):
            continue
        t = raw
        if t and t[0] in ("'", "^"):
            t = t[1:]
        if t.endswith("$"):
            t = t[:-1]
        if t:
            terms.append(t)
    return terms


@cli.command("_highlight", hidden=True)
@click.argument("query")
def _highlight_cmd(query: str) -> None:
    """Read stdin, write stdout with each (case-insensitive) literal
    occurrence of every fzf search term in ``query`` wrapped in bold
    red. Used by the inspect picker's preview pipeline so the user's
    typed query is visible in the preview pane.

    Substring match per term — agrees with fzf's exact-match operator
    (``'word``) and the default fuzzy mode when the fuzzy chars happen
    to be contiguous. Operators ``'``, ``^``, ``$`` are stripped from
    each term before matching; negated terms (``!word``) and ``|`` OR
    separators are skipped (nothing to highlight). Special characters
    in each term are escaped, so typing ``.``, ``[``, etc. is safe.
    """
    import re
    import sys
    terms = _parse_fzf_terms(query)
    if not terms:
        sys.stdout.write(sys.stdin.read())
        return
    # Longest terms first so a longer substring isn't shadowed by a
    # shorter one that's a prefix of it.
    terms.sort(key=len, reverse=True)
    pat = re.compile(
        "|".join(re.escape(t) for t in terms), re.IGNORECASE
    )
    for line in sys.stdin:
        sys.stdout.write(
            pat.sub(lambda m: f"\033[1;31m{m.group(0)}\033[0m", line)
        )


def _render_sse_stream(chunks) -> int:
    """Parse an OpenAI-compatible SSE stream and emit only the generated
    content (and tool-call name/arguments) to stdout. Returns the total
    number of raw bytes consumed so the caller can report it."""
    import json as _json

    buf = ""
    total = 0
    tool_open = False
    for chunk in chunks:
        total += len(chunk)
        buf += chunk.decode("utf-8", errors="replace")
        while True:
            sep = buf.find("\n\n")
            if sep < 0:
                break
            event, buf = buf[: sep], buf[sep + 2:]
            for line in event.split("\n"):
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if not payload or payload == "[DONE]":
                    continue
                try:
                    obj = _json.loads(payload)
                except _json.JSONDecodeError:
                    continue
                delta = (obj.get("choices") or [{}])[0].get("delta") or {}
                content = delta.get("content")
                if content:
                    sys.stdout.write(content)
                    sys.stdout.flush()
                for tc in delta.get("tool_calls") or []:
                    fn = tc.get("function") or {}
                    name = fn.get("name")
                    if name:
                        prefix = ")\n" if tool_open else ""
                        sys.stdout.write(f"{prefix}[tool:{name}](")
                        tool_open = True
                    args = fn.get("arguments")
                    if args:
                        sys.stdout.write(args)
                        sys.stdout.flush()
    if tool_open:
        sys.stdout.write(")\n")
    elif not buf.endswith("\n"):
        sys.stdout.write("\n")
    sys.stdout.flush()
    return total


def _render_buffered_completion(obj: dict) -> None:
    """Render a non-streamed /v1/chat/completions response: just the
    message text, then a compact tool-call summary if any."""
    msg = ((obj.get("choices") or [{}])[0].get("message")) or {}
    content = msg.get("content") or ""
    if content:
        sys.stdout.write(content)
        if not content.endswith("\n"):
            sys.stdout.write("\n")
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        sys.stdout.write(f"[tool:{fn.get('name', '?')}]({fn.get('arguments', '')})\n")
    sys.stdout.flush()


@cli.command("replay")
@click.argument("request_id", required=False, shell_complete=_complete_request_ids)
@click.option(
    "--target",
    required=True,
    help="Base URL of an OpenAI-compatible server (e.g. "
    "http://127.0.0.1:8000). The captured JSON object is POSTed to "
    "<target>/v1/chat/completions with no agentcap-side "
    "normalisation (the original wire whitespace / key ordering is "
    "not preserved by capture, only the JSON object is).",
)
@click.option(
    "--source",
    default=None,
    help="Where to look up the request (see ``agentcap inspect``). "
    "Defaults to scanning the local workspace.",
)
@click.option(
    "--timeout",
    type=float,
    default=600.0,
    show_default=True,
    help="Per-request HTTP timeout in seconds.",
)
@click.option(
    "--raw", "raw_output", is_flag=True,
    help="Dump the raw response bytes (SSE for streamed, JSON for "
    "buffered) instead of rendering just the generated text. Use "
    "when debugging the wire shape.",
)
@click.option(
    "--api-key",
    "api_key",
    envvar="AGENTCAP_API_KEY",
    default=None,
    help="Bearer token forwarded to ``--target``. Required for "
    "authenticated upstreams like the HF Router; when ``--target`` "
    "points at the router we also auto-try ``HF_TOKEN`` and "
    "``~/.cache/huggingface/token`` (mirrors ``agentcap run``).",
)
def replay_cmd(
    request_id: str | None, target: str, source: str | None,
    timeout: float, raw_output: bool, api_key: str | None,
) -> None:
    """Re-issue one captured request to an OpenAI-compatible endpoint.

    Without a request-id, opens the same fzf picker as ``agentcap inspect``
    and replays whatever you select. Single-turn only — multi-turn replay
    diverges as soon as the new model responds differently. The captured
    JSON object is sent without agentcap-side normalisation (captures
    store parsed JSON, so the original byte sequence isn't preserved).
    Streams the rendered generation (assistant text + ``[tool:NAME](args)``
    markers) to stdout; pass ``--raw`` to dump the SSE / JSON response
    bytes from the target instead. Status and timing go to stderr.
    """
    import json as _json
    import time

    import httpx

    if request_id is None:
        if source is not None:
            raise click.UsageError(
                "request-id is required when --source points outside the workspace"
            )
        run = _pick_workspace_run()
        if run is None:
            return
        picked = _pick_workspace_request(run)
        if picked is None:
            return  # cancelled
        request_id = picked
    full_rid, body, _, _, _ = _resolve_request_id(request_id, source)
    url = target.rstrip("/") + "/v1/chat/completions"
    is_stream = bool(body.get("stream"))
    # Resolve auth the same way ``agentcap run`` does: explicit
    # ``--api-key`` / ``AGENTCAP_API_KEY`` wins, otherwise auto-pick
    # up ``HF_TOKEN`` / ~/.cache/huggingface/token when ``--target``
    # looks like the HF Router. Without this, replaying a captured
    # request against the router returned 401 every time.
    api_key, api_key_source = _resolve_api_key(
        upstream=target, explicit_api_key=api_key,
    )
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    if api_key_source:
        click.echo(
            f"  [auth] token source={api_key_source}",
            err=True,
        )
    # Rough input size — chars/4 is a coarse token estimate but it sets
    # expectations: 15k tokens means "wait a minute," not "something's wrong".
    msgs_chars = sum(
        len(_json.dumps(m, ensure_ascii=False))
        for m in (body.get("messages") or [])
    )
    click.echo(
        f"  rid={full_rid} POST {url} "
        f"({'streaming' if is_stream else 'buffered'}, "
        f"messages={len(body.get('messages') or [])}, "
        f"tools={len(body.get('tools') or [])}, "
        f"~{msgs_chars // 4} input tokens)…",
        err=True,
    )
    t0 = time.monotonic()
    n_bytes = 0
    status: int | None = None
    try:
        if is_stream:
            # Stream chunks to stdout as they arrive so a long generation
            # gives immediate feedback instead of a wall of silence.
            with httpx.stream(
                "POST", url, json=body, timeout=timeout, headers=headers,
            ) as resp:
                status = resp.status_code
                click.echo(
                    f"  ← headers status={status} content-type="
                    f"{resp.headers.get('content-type', '?')}",
                    err=True,
                )
                if raw_output or status != 200:
                    # Errors come back as a single JSON object, not SSE.
                    # Always dump them raw so the user sees the message.
                    for chunk in resp.iter_raw():
                        sys.stdout.buffer.write(chunk)
                        sys.stdout.buffer.flush()
                        n_bytes += len(chunk)
                else:
                    n_bytes = _render_sse_stream(resp.iter_raw())
        else:
            resp = httpx.post(url, json=body, timeout=timeout, headers=headers)
            status = resp.status_code
            n_bytes = len(resp.content)
            if raw_output or status != 200:
                try:
                    click.echo(_json.dumps(resp.json(), indent=2, ensure_ascii=False))
                except ValueError:
                    sys.stdout.buffer.write(resp.content)
                    sys.stdout.buffer.write(b"\n")
            else:
                _render_buffered_completion(resp.json())
    except httpx.HTTPError as exc:
        raise click.ClickException(f"replay failed: {exc}")
    dt = time.monotonic() - t0
    click.echo(
        f"  status={status} duration={dt:.2f}s bytes={n_bytes}",
        err=True,
    )


def main() -> int:
    cli.main(standalone_mode=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
