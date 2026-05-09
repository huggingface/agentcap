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
    help="Model id passed to the agent (required for opencode/goose/pi; "
    "ignored by hermes since hermes resolves the model from its own config).",
)
@click.option(
    "--upstream",
    required=True,
    help="Base URL of the upstream model server (e.g. http://127.0.0.1:8000).",
)
@click.option(
    "--listen",
    default="127.0.0.1:8001",
    show_default=True,
    help="HOST:PORT the in-process proxy binds on. Point your agent at this.",
)
@click.option(
    "--workdir",
    required=True,
    type=click.Path(file_okay=False, dir_okay=True),
    help="Output directory for traces, sessions logs, and the run summary.",
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
    listen: str,
    workdir: str,
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
    from .proxy import serve_in_thread

    workdir_p = Path(workdir)
    traces = workdir_p / "traces"
    sessions = workdir_p / "sessions"
    sandbox = workdir_p / "sandbox"
    traces.mkdir(parents=True, exist_ok=True)
    sessions.mkdir(parents=True, exist_ok=True)
    # Persist the agent name so `agentcap export` can tag bucket-pushed
    # parquet filenames with it. The capture proxy itself stays
    # agent-agnostic; this is purely an orchestrator hint.
    (traces / "_meta.json").write_text(json.dumps({"agent": agent}))
    # Empty sandbox dir = the agent's cwd. Keeps cwd-side state (e.g.
    # Hermes auto-injecting AGENTS.md / CLAUDE.md from cwd into every
    # system prompt) from leaking into the trace.
    sandbox.mkdir(parents=True, exist_ok=True)

    host, port = _parse_listen(listen)

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

    proxy_base_url = f"http://{host}:{port}/v1"
    driver_kwargs: dict = {}
    if agent == "hermes":
        # Build a temp HERMES_HOME so requests are routed through the
        # capture proxy without modifying the user's ~/.hermes/config.yaml.
        # Run hermes from the sandbox dir so cwd-side state (AGENTS.md,
        # CLAUDE.md, .cursorrules) doesn't leak into captured prompts.
        driver_kwargs["proxy_base_url"] = proxy_base_url
        driver_kwargs["cwd"] = sandbox
    elif agent in ("opencode", "goose", "pi"):
        if not model:
            raise click.UsageError(
                f"--model is required for --agent {agent}"
            )
        driver_kwargs["proxy_base_url"] = proxy_base_url
        driver_kwargs["model"] = model
        driver_kwargs["cwd"] = sandbox
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
