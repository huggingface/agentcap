use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "agentcap",
    version,
    about = "Capture agent ↔ model interactions and publish them as HF datasets."
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Drive an agent through a corpus, capturing every chat-completion.
    Run {
        /// Agent driver: hermes | opencode | goose | pi.
        #[arg(long)]
        agent: String,
        /// Model id the agent requests (recorded as the `model` field).
        #[arg(long)]
        model: Option<String>,
        /// Base URL of the upstream model server.
        #[arg(long)]
        upstream: String,
        /// Bearer token forwarded upstream (env: AGENTCAP_API_KEY).
        #[arg(long, env = "AGENTCAP_API_KEY")]
        api_key: Option<String>,
        /// Host dir exposed as the agent's cwd (bind-mounted writable).
        #[arg(long)]
        sandbox: Option<String>,
        /// Host dir with a huggingface/skills checkout (bind-mounted read-only).
        #[arg(long)]
        skills: Option<String>,
        /// Plain-text file: one prompt per line (# comments + blanks ignored).
        #[arg(long)]
        tasks: String,
        /// Total turns per task (1 = no follow-ups).
        #[arg(long, default_value_t = 1)]
        turns: i64,
        /// Follow-up strategy for turns 2..N: continue | templates | synthesized.
        #[arg(long, default_value = "continue")]
        followup: String,
        /// Per-turn timeout in seconds.
        #[arg(long, default_value_t = 1200.0)]
        timeout: f64,
    },
    /// List runs in a workspace.
    Ls {
        /// Workspace dir (defaults to ./.agentcap).
        workspace: Option<String>,
        /// Include upstream + per-run counts.
        #[arg(short = 'l', long)]
        long: bool,
    },
    /// Render captures to parquet and push to the Hub.
    Export {
        /// Run ids or capture dirs (omit with --all).
        targets: Vec<String>,
        #[arg(long)]
        all: bool,
        /// owner/base destination (required).
        #[arg(long)]
        push: String,
        /// Skip the trufflehog secret scan.
        #[arg(long)]
        no_scan: bool,
    },
    /// Browse captured requests in a terminal picker.
    Inspect {
        /// run-id, capture dir, .parquet, or hf://datasets/<owner>/<name>.
        target: Option<String>,
        /// Print only the resolved request-id (for a hex rid target).
        #[arg(long)]
        rid: bool,
    },
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Run {
            agent,
            model,
            upstream,
            api_key,
            sandbox,
            skills,
            tasks,
            turns,
            followup,
            timeout,
        } => agentcap::run::run(
            agent, model, upstream, api_key, sandbox, skills, tasks, turns, followup, timeout,
        ),
        Cmd::Ls { workspace, long } => agentcap::ls::run(workspace, long),
        Cmd::Export {
            targets,
            all,
            push,
            no_scan,
        } => agentcap::export::run(targets, all, push, no_scan),
        Cmd::Inspect { target, rid } => agentcap::inspect::run(target, rid),
    }
}
