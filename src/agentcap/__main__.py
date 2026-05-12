"""CLI entrypoint.

Subcommands:
- ``agentcap proxy``  — run the capture proxy in front of a model server.
- ``agentcap export`` — render a captured trace dir into a JSONL dataset.
- ``agentcap run``    — drive an agent CLI through a corpus.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

from . import __version__
from .drivers import known_drivers as _known_drivers


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="agentcap")
def cli() -> None:
    """agentcap: capture LLM-agent chat-completion traces."""


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


@cli.command("proxy")
@click.option(
    "--upstream",
    required=True,
    help="Base URL of the model server (e.g. http://127.0.0.1:8000).",
)
@click.option(
    "--listen",
    default="127.0.0.1:8001",
    show_default=True,
    help="HOST:PORT to bind the proxy on.",
)
@click.option(
    "--trace-dir",
    required=True,
    type=click.Path(file_okay=False, dir_okay=True, writable=True),
    help="Directory to write captured request/response JSON files into.",
)
def proxy_cmd(upstream: str, listen: str, trace_dir: str) -> None:
    """Run the capture proxy in front of a model server."""
    from .proxy import serve

    host, port = _parse_listen(listen)
    click.echo(
        f"agentcap proxy: forwarding {host}:{port} -> {upstream}, "
        f"capturing to {trace_dir}",
        err=True,
    )
    serve(upstream=upstream, trace_dir=trace_dir, host=host, port=port)


@cli.command("export")
@click.argument(
    "trace_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
)
@click.option(
    "--model",
    default=None,
    help="HF model id whose chat template to render with. "
    "If omitted, inferred from the captured request bodies "
    "(fails if they're not all for the same model).",
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False),
    default=None,
    help="Local parquet file to write.",
)
@click.option(
    "--push",
    default=None,
    help="Storage Bucket URI to push to: hf://buckets/<owner>/<name>[/<prefix>]. "
    "Dataset repos aren't accepted — render to --output and use `hf upload` instead.",
)
@click.option(
    "--agent",
    default=None,
    help="Agent name to embed in the bucket filename (so pushes are "
    "grouped by agent + model). If omitted, auto-detected from "
    "<trace-dir>/_meta.json (written by `agentcap run`); falls back "
    "to no agent tag for trace dirs produced by other flows.",
)
@click.option(
    "--workers",
    type=int,
    default=1,
    show_default=True,
    help="Render rows in a process pool of this size. >1 parallelises "
    "the per-row chat-template render (CPU-bound). Each worker "
    "re-loads the tokenizer on init.",
)
def export_cmd(
    trace_dir: str,
    model: str | None,
    output: str | None,
    push: str | None,
    agent: str | None,
    workers: int,
) -> None:
    """Render a captured trace dir into a parquet dataset."""
    from .export import (
        _BUCKET_PREFIX,
        detect_agent,
        detect_model,
        export_local,
        load_processor,
        push_bucket,
    )

    if not output and not push:
        raise click.UsageError("one of --output or --push is required")
    if output and push:
        raise click.UsageError("pass --output OR --push, not both")
    if push and not push.startswith(_BUCKET_PREFIX):
        raise click.UsageError(
            f"--push only accepts bucket URIs ({_BUCKET_PREFIX}<owner>/<name>...). "
            f"For Dataset repos, render to --output and use `hf upload`."
        )

    # Always run detection — its job is to enforce that the trace dir is
    # for exactly one model. --model can fall back to the detected value
    # but cannot bypass the uniqueness check.
    try:
        detected = detect_model(trace_dir)
    except ValueError as exc:
        raise click.UsageError(str(exc))

    if model is None:
        if detected is None:
            raise click.UsageError(
                f"trace dir {trace_dir} has no captured requests with a "
                f"model field; pass --model explicitly."
            )
        model = detected
        click.echo(
            f"agentcap export: using model {model!r} (auto-detected)",
            err=True,
        )

    if agent is None:
        agent = detect_agent(trace_dir)
        if agent is not None:
            click.echo(
                f"agentcap export: using agent {agent!r} (auto-detected)",
                err=True,
            )

    proc = load_processor(model)

    if output:
        n_rows = export_local(
            trace_dir, output, processor=proc, model=model, workers=workers
        )
        click.echo(
            f"agentcap export: wrote {n_rows} rows to {output}", err=True
        )
    else:
        n_rows = push_bucket(
            trace_dir, push, processor=proc, model=model, agent=agent,
            workers=workers,
        )
        click.echo(
            f"agentcap export: wrote {n_rows} rows to bucket {push}", err=True
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
    "that made captured traces lie about which model was actually run.",
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
    "Falls back to the AGENTCAP_API_KEY env var.",
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
    required=True,
    type=click.Path(file_okay=False, dir_okay=True),
    help="Output directory for traces, sessions logs, and the run summary.",
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
    help="Synthesizer endpoint (only used with --followup synthesized). Should bypass the capture proxy.",
)
@click.option(
    "--synth-model",
    default=None,
    help="Synthesizer model id (only used with --followup synthesized).",
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
    """Drive an agent CLI through a corpus, capture traces, summarise."""
    import json

    from .drivers import get_driver
    from .followups import get_followup
    from .orchestrator import Orchestrator, read_tasks_txt
    from .proxy import (
        IN_PROCESS_PROXY_HOST,
        IN_PROCESS_PROXY_PORT,
        serve_in_thread,
    )
    from .sandbox import require_sandbox_or_die

    # --- validation: everything that can reject the CLI invocation
    # goes here, BEFORE we touch the sandbox or the workdir. So a
    # bad CLI call doesn't pay the cost of building/booting the
    # per-agent image/VM or leave stray temp dirs behind.

    # In-process proxy bind address. ``--listen HOST:PORT`` overrides
    # the default for hosts where it collides. The image entrypoint
    # reads AGENTCAP_PROXY_URL and patches each agent's config at run
    # time. When the bind host is 0.0.0.0 (or otherwise non-routable
    # from inside the sandbox), advertise a sandbox-reachable host
    # instead: ``host.lima.internal`` on macOS / Lima, ``127.0.0.1``
    # on Linux / bwrap.
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

    if followup == "synthesized":
        if not synth_upstream or not synth_model:
            raise click.UsageError(
                "--followup synthesized requires --synth-upstream and --synth-model"
            )
        fu = get_followup(
            "synthesized", upstream=synth_upstream, model=synth_model
        )
    else:
        fu = get_followup(followup)

    if not model:
        raise click.UsageError(
            f"--model is required for --agent {agent}"
        )

    # --- sandbox setup: from here on, side effects.

    def _sb_log(msg: str) -> None:
        click.echo(f"  [sandbox] {msg}", err=True)

    # Builds the per-agent image (Linux) or boots the per-agent VM
    # (macOS) before returning. First call per agent can take minutes.
    # ``AGENTCAP_PROXY_URL`` is read by the image's entrypoint script
    # to render the agent's config files at startup;
    # ``AGENTCAP_SKILLS_DIR`` (when --skills is set) tells the same
    # script where the bind-mounted skills checkout lives so it can
    # symlink into the agent-specific discovery location.
    sandbox_env = {"AGENTCAP_PROXY_URL": proxy_url, "AGENTCAP_MODEL": model}
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

    workdir_p = Path(workdir)
    traces = workdir_p / "traces"
    sessions = workdir_p / "sessions"
    traces.mkdir(parents=True, exist_ok=True)
    sessions.mkdir(parents=True, exist_ok=True)
    # Persist the agent name so `agentcap export` can tag bucket-pushed
    # parquet filenames with it. The capture proxy itself stays
    # agent-agnostic; this is purely an orchestrator hint.
    (traces / "_meta.json").write_text(json.dumps({"agent": agent}))
    # Agent cwd resolution. If the user passed --workspace, use that
    # host path (bind-mounted into the sandbox; the agent sees its
    # contents, e.g. a transformers worktree). Otherwise mint a fresh
    # hermetic temp dir so cwd-side state (Hermes auto-injecting
    # AGENTS.md, etc.) doesn't leak between runs.
    if workspace is not None:
        sandbox_cwd = str(Path(workspace).resolve())
    else:
        sandbox_cwd = sandbox.mkdtemp(prefix="agentcap-run-")

    # The agent talks to the in-process proxy at the fixed URL baked
    # into the per-agent image — drivers no longer need a base_url
    # argument. ``cwd`` is the per-run sandbox-side dir we just
    # minted; the orchestrator runs the agent from there so cwd-side
    # state (Hermes auto-injecting AGENTS.md, etc.) is isolated.
    driver_kwargs: dict = {"sandbox": sandbox, "cwd": sandbox_cwd}
    if agent in ("opencode", "goose", "pi"):
        driver_kwargs["model"] = model
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
        with serve_in_thread(upstream, traces, host=host, port=port):
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


def main() -> int:
    cli.main(standalone_mode=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
