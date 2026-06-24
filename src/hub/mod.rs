//! Hugging Face Hub access (synchronous, via hf-hub's blocking API).
//!
//! Token discovery is lifted from funes; the rest are thin helpers over
//! `HFClientSync` / `HFRepositorySync<RepoTypeDataset>`: list a dataset's parquet
//! files, download one, ranged-read another (footer picker), and resolve request
//! bodies from a dataset (the `hf://` source for `captures`).

pub mod collection;
pub mod footer;

use std::collections::{HashMap, HashSet};
use std::ops::Range;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use bytes::Bytes;
use hf_hub::repository::RepoTreeEntry;
use hf_hub::{HFClient, HFClientSync, HFRepositorySync, RepoTypeDataset};
use serde_json::Value;

pub use hf_hub::repository::{CommitInfo, CommitOperation};
pub use hf_hub::RepoTypeDataset as DatasetRepo;

pub type DatasetHandle = HFRepositorySync<RepoTypeDataset>;

/// HF token from the standard env vars, else the cached token file. Same
/// precedence as funes: `HF_TOKEN` → `HUGGING_FACE_HUB_TOKEN` →
/// `HUGGINGFACE_TOKEN` → `~/.cache/huggingface/token`.
pub fn hf_token() -> Option<String> {
    let token_file = std::env::var("HOME")
        .ok()
        .map(|h| PathBuf::from(h).join(".cache/huggingface/token"));
    token_from(|k| std::env::var(k).ok(), token_file.as_deref())
}

fn token_from(env: impl Fn(&str) -> Option<String>, token_file: Option<&Path>) -> Option<String> {
    for var in ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"] {
        if let Some(t) = env(var) {
            let t = t.trim();
            if !t.is_empty() {
                return Some(t.to_string());
            }
        }
    }
    let cached = std::fs::read_to_string(token_file?).ok()?;
    let t = cached.trim();
    (!t.is_empty()).then(|| t.to_string())
}

/// Build a blocking HF client, authenticated if a token is discoverable.
pub fn client_sync() -> Result<HFClientSync> {
    let mut builder = HFClient::builder();
    if let Some(token) = hf_token() {
        builder = builder.token(token);
    }
    builder.build_sync().map_err(|e| anyhow!("building HF client: {e}"))
}

/// Split `"owner/name"`, rejecting anything else.
pub fn split_repo_id(repo_id: &str) -> Result<(String, String)> {
    let parts: Vec<&str> = repo_id.split('/').collect();
    if parts.len() != 2 || parts[0].is_empty() || parts[1].is_empty() {
        bail!("repo id must be <owner>/<name>, got {repo_id:?}");
    }
    Ok((parts[0].to_string(), parts[1].to_string()))
}

/// A dataset repo handle on a fresh client.
pub fn dataset(repo_id: &str) -> Result<DatasetHandle> {
    let (owner, name) = split_repo_id(repo_id)?;
    Ok(client_sync()?.dataset(owner, name))
}

/// `data/*.parquet` files in a dataset, as `(path, size)`, sorted by path.
pub fn list_parquet_files(repo: &DatasetHandle) -> Result<Vec<(String, u64)>> {
    let entries = repo
        .list_tree()
        .recursive(true)
        .expand(true)
        .send()
        .context("listing dataset tree")?;
    let mut out: Vec<(String, u64)> = entries
        .into_iter()
        .filter_map(|e| match e {
            RepoTreeEntry::File { path, size, .. } if path.starts_with("data/") && path.ends_with(".parquet") => {
                Some((path, size))
            }
            _ => None,
        })
        .collect();
    out.sort();
    Ok(out)
}

/// Paths of every file in a dataset (for README-already-present checks).
pub fn list_files(repo: &DatasetHandle) -> Result<Vec<String>> {
    let entries = repo
        .list_tree()
        .recursive(true)
        .send()
        .context("listing dataset tree")?;
    Ok(entries
        .into_iter()
        .filter_map(|e| match e {
            RepoTreeEntry::File { path, .. } => Some(path),
            _ => None,
        })
        .collect())
}

/// Download a whole file into the HF cache, returning its local path.
pub fn download_file(repo: &DatasetHandle, path: &str) -> Result<PathBuf> {
    repo.download_file()
        .filename(path)
        .send()
        .with_context(|| format!("downloading {path}"))
}

/// Ranged read of a file (start-inclusive, end-exclusive) — the footer picker.
pub fn download_range(repo: &DatasetHandle, path: &str, range: Range<u64>) -> Result<Bytes> {
    repo.download_file_to_bytes()
        .filename(path)
        .range(range)
        .send()
        .with_context(|| format!("ranged read of {path}"))
}

/// The `hf://` source for `captures`: scan parquets under `data/` until every
/// wanted request id is found.
pub fn load_request_bodies_from_dataset(repo_id: &str, wanted: &HashSet<String>) -> Result<HashMap<String, Value>> {
    let repo = dataset(repo_id)?;
    let files = list_parquet_files(&repo)?;
    let mut out = HashMap::new();
    let mut remaining = wanted.clone();
    for (path, _) in files {
        let local = download_file(&repo, &path)?;
        let found = crate::parquet_io::read_request_bodies(&local, &remaining)?;
        for (k, v) in found {
            remaining.remove(&k);
            out.insert(k, v);
        }
        if remaining.is_empty() {
            break;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn token_env_beats_file_and_trims() {
        let env: HashMap<&str, &str> = [("HF_TOKEN", "  hf_envtok \n")].into_iter().collect();
        let file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(file.path(), "hf_filetok").unwrap();
        let got = token_from(|k| env.get(k).map(|s| s.to_string()), Some(file.path()));
        assert_eq!(got.as_deref(), Some("hf_envtok"));
    }

    #[test]
    fn token_falls_back_to_file() {
        let file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(file.path(), "  hf_filetok\n").unwrap();
        assert_eq!(token_from(|_| None, Some(file.path())).as_deref(), Some("hf_filetok"));
    }

    #[test]
    fn token_blank_env_skipped_none_without_file() {
        let env: HashMap<&str, &str> = [("HF_TOKEN", "   ")].into_iter().collect();
        assert_eq!(token_from(|k| env.get(k).map(|s| s.to_string()), None), None);
    }

    #[test]
    fn split_repo_id_validates() {
        assert_eq!(split_repo_id("acme/kb").unwrap(), ("acme".into(), "kb".into()));
        assert!(split_repo_id("acme").is_err());
        assert!(split_repo_id("a/b/c").is_err());
        assert!(split_repo_id("/kb").is_err());
    }
}
