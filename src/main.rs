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
