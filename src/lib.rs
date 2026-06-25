//! agentcap — capture agent ↔ model interactions and publish them as HF datasets.
//!
//! `run` drives an agent through a corpus behind a capture proxy; `export` renders
//! the captures to parquet and pushes them to the Hub; `inspect` / `ls` browse them.

pub mod captures;
pub mod diff;
pub mod drivers;
pub mod export;
pub mod followups;
pub mod hub;
pub mod inspect;
pub mod ls;
pub mod model;
pub mod orchestrator;
pub mod parquet_io;
pub mod provider;
pub mod proxy;
pub mod query;
pub mod run;
pub mod sandbox;
pub mod scan;
pub mod sse;
