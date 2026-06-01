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


def _is_hf_router_upstream(upstream: str) -> bool:
    host = (urlparse(upstream).hostname or "").lower()
    return host == "router.huggingface.co"


def _is_lima_host() -> bool:
    import platform
    import shutil
    return platform.system() == "Darwin" and shutil.which("limactl") is not None


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

    if followup == "synthesized":
        fu = get_followup("synthesized", upstream=upstream, model=model)
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

    # Resolve --sandbox up front: it joins the per-VM mount set
    # alongside --skills (RO) and the traces dir (RW), so the Lima
    # backend can configure all three at provision time.
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

    # Lima VM sees 127.0.0.1 as its own loopback — bind on all ifs and
    # advertise the host bridge instead.
    bind_host, agent_host = (
        ("0.0.0.0", "host.lima.internal") if _is_lima_host()
        else ("127.0.0.1", "127.0.0.1")
    )
    with serve_in_thread(upstream, captures, host=bind_host) as proxy:
        proxy_url = f"http://{agent_host}:{proxy.port}/v1"
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
) -> tuple[str, dict, dict | None]:
    """Resolve ``rid`` (full or short prefix) to ``(full_rid, body,
    response_record)``.

    - If ``source`` is given, looks the rid up there via
      ``replay.load_request`` (any agentcap-supported source: dir,
      parquet, hf://) — exact match only. Response record is
      unavailable in that path.
    - Otherwise scans the workspace, accepting a prefix (git-style)
      and returning both the request body and the paired response.
    """
    from . import replay

    if source is not None:
        try:
            return rid, replay.load_request(source, rid), None
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
    return full_rid, body, resp_rec


def _enumerate_workspace_requests(scope: str | None) -> list[dict]:
    """Walk captures across the workspace (or one run if ``scope`` is a
    run-id) and return one row per captured request, in chronological
    order. Each row has ``run_id``, ``rid``, ``captured_at``, ``status``,
    and ``preview`` (last user message, truncated)."""
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
        for req_path in captures.glob("*.request.json"):
            rid = req_path.stem.split(".")[0]
            try:
                req = _json.loads(req_path.read_text())
            except (OSError, _json.JSONDecodeError):
                continue
            resp_path = captures / f"{rid}.response.json"
            status = "?"
            if resp_path.is_file():
                try:
                    status = str(_json.loads(resp_path.read_text()).get("status_code", "?"))
                except (OSError, _json.JSONDecodeError):
                    pass
            messages = (req.get("body") or {}).get("messages") or []
            last_user = next(
                (m.get("content") for m in reversed(messages) if m.get("role") == "user"),
                "",
            )
            if isinstance(last_user, list):
                # Tool-result / multi-modal message content can be a list.
                last_user = " ".join(
                    p.get("text", "") for p in last_user if isinstance(p, dict)
                )
            preview = (last_user or "").replace("\n", " ").strip()
            rows.append({
                "run_id": run_dir.name,
                "rid": rid,
                "captured_at": int(req.get("captured_at", 0)),
                "status": status,
                "preview": preview,
            })
    rows.sort(key=lambda r: (r["run_id"], r["captured_at"]))
    return rows


def _format_inspect_rows(
    rows: list[dict], *, include_run: bool
) -> tuple[str, list[str]]:
    """Format rows as a header line + body lines. Rid first (it's what
    you copy into replay), then time/status/preview, with run-id last
    only when browsing across multiple runs. The task preview column
    shrinks to fit the terminal so the task is always visible."""
    import shutil
    import time

    cols = ["RID", "TIME", "STATUS", "PREVIEW"]
    if include_run:
        cols.insert(3, "RUN")
    widths = {c: len(c) for c in cols}
    widths["RID"] = max(widths["RID"], 8)  # git-style 8-char prefix
    widths["TIME"] = max(widths["TIME"], 8)
    widths["STATUS"] = max(widths["STATUS"], 3)
    if include_run:
        widths["RUN"] = max(
            widths["RUN"],
            max((len(r["run_id"]) for r in rows), default=0),
        )

    # Compute the budget for the task preview column from the terminal
    # width. fzf's preview pane is right:60%, so the row gets the left
    # 40% — use that when we're going through fzf (heuristic: caller
    # passes us the same rows). For the plain-table fallback, use the
    # full width. We err on the wider side since fzf wraps gracefully.
    term_w = shutil.get_terminal_size((120, 24)).columns
    sep_w = 2 * (len(cols) - 1)
    fixed_w = sum(widths[c] for c in cols if c != "PREVIEW") + sep_w
    preview_w = max(20, term_w - fixed_w)
    widths["PREVIEW"] = preview_w

    def _trim(s: str, w: int) -> str:
        if len(s) <= w:
            return s
        return s[: max(1, w - 1)] + "…"

    def _fmt(rid, ts, status, run, preview) -> str:
        cells = [
            f"{rid:<{widths['RID']}}",
            f"{ts:<{widths['TIME']}}",
            f"{status:>{widths['STATUS']}}",
        ]
        if include_run:
            cells.append(f"{run:<{widths['RUN']}}")
        cells.append(_trim(preview, preview_w))
        return "  ".join(cells)

    header = _fmt("RID", "TIME", "STATUS", "RUN", "PREVIEW")
    body = [
        _fmt(
            r["rid"][:8],
            time.strftime("%H:%M:%S", time.gmtime(r["captured_at"]))
            if r["captured_at"] else "?",
            r["status"],
            r["run_id"],
            r["preview"],
        )
        for r in rows
    ]
    return header, body


def _fzf_pick(
    header: str, lines: list[str], preview_cmd: str
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
    proc = subprocess.run(
        [
            "fzf",
            "--ansi",
            "--header", header,
            "--header-first",
            "--preview", preview_cmd,
            "--preview-window=right:60%:wrap",
            "--no-sort",
        ],
        input="\n".join(lines),
        capture_output=True,
        text=True,
    )
    picked = proc.stdout.rstrip("\n") if proc.returncode == 0 else ""
    return (picked or None, True)


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
    header, lines = _format_inspect_rows(rows, include_run=include_run)
    preview = (
        f"{sys.executable} -m agentcap _preview {{1}} 2>/dev/null | head -400"
    )

    picked, fzf_available = _fzf_pick(header, lines, preview)
    if not fzf_available:
        click.echo(header)
        for line in lines:
            click.echo(line)
        return None
    if picked is None:
        return None  # cancelled
    return picked.split()[0]


@cli.command("inspect")
@click.argument("target", required=False, shell_complete=_complete_request_ids)
@click.option(
    "--source",
    default=None,
    help="Where to look up the request: a capture dir, a .parquet, or "
    "hf://datasets/<owner>/<name>[/<subdir>]. Only honored when "
    "TARGET is a request-id; defaults to scanning the local workspace.",
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
    - ``agentcap inspect``              browse all workspace requests (fzf picker)
    - ``agentcap inspect <run-id>``     browse one run
    - ``agentcap inspect <rid>``        print the captured body for that request

    A rid is 32 hex chars (proxy-minted UUID); a run-id contains a dash.
    Falls back to a plain table when fzf is not on PATH.
    """
    import json as _json

    # rid (full or short prefix) → body dump (single request).
    if target and "-" not in target:
        full_rid, body, resp_rec = _resolve_request_id(target, source)
        if resp_rec is not None:
            click.echo(
                f"  request_id={full_rid} "
                f"captured_at={resp_rec.get('captured_at_resp', '?')} "
                f"status={resp_rec.get('status_code', '?')}",
                err=True,
            )
        click.echo(_json.dumps(body, indent=2, ensure_ascii=False))
        return

    # No arg or run-id → enumerate + pick.
    pick = _pick_workspace_request(target)
    if pick is None:
        return  # cancelled or no-fzf table-only path
    full_rid, body, resp_rec = _resolve_request_id(pick, None)
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


@cli.command("_preview", hidden=True)
@click.argument("request_id")
def _preview_cmd(request_id: str) -> None:
    """Internal: dual TASK + REQUEST view used by the fzf preview pane.

    Not part of the public CLI surface — hidden from ``--help``. The
    user-facing inspector is ``agentcap inspect <rid>``.
    """
    import json as _json

    full_rid, body, resp_rec = _resolve_request_id(request_id, None)
    messages = body.get("messages") or []
    last_user = next(
        (m.get("content") for m in reversed(messages) if m.get("role") == "user"),
        "",
    )
    if isinstance(last_user, list):
        last_user = " ".join(
            p.get("text", "") for p in last_user if isinstance(p, dict)
        )
    status = (
        resp_rec.get("status_code") if resp_rec is not None else "?"
    )
    click.echo(f"rid:    {full_rid}")
    click.echo(f"status: {status}")
    click.echo(f"model:  {body.get('model', '?')}")
    click.echo()
    click.echo("─── TASK ────────────────────────────────────────────────")
    click.echo(last_user or "(no user message)")
    click.echo()
    click.echo("─── REQUEST ─────────────────────────────────────────────")
    click.echo(_json.dumps(body, indent=2, ensure_ascii=False))


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
    "http://127.0.0.1:8000). The body is POSTed verbatim to "
    "<target>/v1/chat/completions.",
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
    diverges as soon as the new model responds differently (see
    ROADMAP.md). The body is sent byte-faithfully; no normalisation.
    Prints the response JSON to stdout and status / timing to stderr.
    """
    import json as _json
    import time

    import httpx

    if request_id is None:
        if source is not None:
            raise click.UsageError(
                "request-id is required when --source points outside the workspace"
            )
        picked = _pick_workspace_request(None)
        if picked is None:
            return  # cancelled or no-fzf table-only path (already printed)
        request_id = picked
    full_rid, body, _ = _resolve_request_id(request_id, source)
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
