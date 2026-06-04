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
        click.echo(_no_workspace_msg(root), err=True)
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
            })
    rows.sort(key=lambda r: (r["run_id"], r["captured_at"]))
    return rows


def _format_inspect_rows(
    rows: list[dict], *, include_run: bool
) -> tuple[str, list[str], list[str]]:
    """Flat table: every row is one captured call, every visible column
    is something the user might fuzzy-filter on (LOC =
    ``task_id.<req_index>``, RID, optional RUN, MESSAGES — the latter
    a ``(+N)`` / ``(init N)`` / ``(-X +Y)`` delta plus a one-line
    role-aware summary). Time / status / model / size live in the
    fzf preview pane (see ``_preview_cmd``).

    Returns ``(header, display_lines, fzf_lines)``. ``display_lines``
    are what the no-fzf table prints. ``fzf_lines`` are the same
    visible content followed by a tab and two metadata fields —
    the full rid and the previous capture's rid — so the fzf preview
    command can pull both via ``{2}`` and ``{3}`` substitution without
    rescanning the capture dir for every hover."""

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

    display: list[str] = []
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
        display.append(line)
        fzf.append(f"{line}\t{r['rid']}\t{r.get('prev_rid') or '-'}")
    return header, display, fzf


def _fzf_pick(
    header: str,
    lines: list[str],
    preview_cmd: str,
    *,
    extra_args: Sequence[str] = (),
) -> tuple[str | None, bool]:
    """Run fzf over ``lines`` with ``header`` pinned at the top. Returns
    ``(picked, available)``:
      - ``(line, True)``  picked from fzf
      - ``(None, True)``  fzf ran, user cancelled (Esc / Ctrl-C)
      - ``(None, False)`` fzf not on PATH
    """
    import shutil
    import subprocess

    # ``AGENTCAP_NO_FZF=1`` lets users compare the fzf and plain-table
    # UX without uninstalling fzf — pretend it isn't there.
    if os.environ.get("AGENTCAP_NO_FZF") or not shutil.which("fzf"):
        return None, False
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
    picked = proc.stdout.rstrip("\n") if proc.returncode == 0 else ""
    return (picked or None, True)


def _pick_workspace_run() -> str | None:
    """Open an fzf picker over the runs in the workspace, returning the
    selected run-id. Without fzf: prints the table and returns None
    (caller treats as cancellation)."""
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

    picked, fzf_available = _fzf_pick(header, lines, preview)
    if not fzf_available:
        click.echo(header)
        for line in lines:
            click.echo(line)
        return None
    if picked is None:
        return None
    tokens = picked.split()
    return tokens[0] if tokens else None


def _pick_workspace_request(scope: str | None) -> str | None:
    """Interactive picker (fzf when available, plain table otherwise) for
    a workspace request. Returns the picked short rid, or ``None`` if
    cancelled / fzf unavailable. In the fzf-missing case the table is
    printed to stdout and ``None`` returned — callers should treat that
    as "user must re-invoke with an explicit rid"."""
    import sys

    rows = _enumerate_workspace_requests(scope)
    if not rows:
        where = f"run {scope!r}" if scope else "workspace"
        raise click.UsageError(f"no captured requests in {where}")

    include_run = scope is None
    header, display, fzf_lines = _format_inspect_rows(rows, include_run=include_run)
    # Tab-delim hidden columns: 2 = full rid, 3 = previous-capture rid
    # (or "-" for the first capture of a task). Pre-computing the prev
    # rid here lets the preview pane skip a full cap-dir rescan per
    # fzf hover.
    preview = (
        f"{sys.executable} -m agentcap _preview {{2}} {{3}} 2>/dev/null | head -400"
    )

    picked, fzf_available = _fzf_pick(
        header, fzf_lines, preview,
        extra_args=["--delimiter", "\t", "--with-nth", "1"],
    )
    if not fzf_available:
        click.echo(header)
        for line in display:
            click.echo(line)
        return None
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
    Falls back to a plain table when fzf is not on PATH.
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

    # No arg → pick a run first; then drill into its calls.
    if target is None:
        target = _pick_workspace_run()
        if target is None:
            return  # cancelled / no fzf table fallback already printed
    pick = _pick_workspace_request(target)
    if pick is None:
        return  # cancelled or no-fzf table-only path
    full_rid, body, resp_rec, _, _ = _resolve_request_id(pick, None)
    if print_rid_only:
        click.echo(full_rid)
        return
    if resp_rec is not None:
        click.echo(
            f"  request_id={full_rid} "
            f"captured_at={resp_rec.get('captured_at_resp', '?')} "
            f"status={resp_rec.get('status_code', '?')}",
            err=True,
        )
    click.echo(_json.dumps(body, indent=2, ensure_ascii=False))


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
def replay_cmd(
    request_id: str | None, target: str, source: str | None,
    timeout: float, raw_output: bool,
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
            return  # cancelled or no-fzf table-only path (already printed)
        request_id = picked
    full_rid, body, _, _, _ = _resolve_request_id(request_id, source)
    url = target.rstrip("/") + "/v1/chat/completions"
    is_stream = bool(body.get("stream"))
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
            with httpx.stream("POST", url, json=body, timeout=timeout) as resp:
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
            resp = httpx.post(url, json=body, timeout=timeout)
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
