//! HF Collections via the raw REST API — hf-hub has no Collections support.
//!
//! Best-effort, mirroring `export.ensure_collection`: find-or-create the
//! `<owner>/<base>` collection and add each dataset as an item. Item-add
//! failures are swallowed (the README cross-links already make the relationship
//! discoverable); only a hard failure to find-or-create surfaces.

use anyhow::{Context, Result};
use serde_json::{json, Value};

use super::hf_token;

fn endpoint() -> String {
    std::env::var("HF_ENDPOINT").unwrap_or_else(|_| "https://huggingface.co".to_string())
}

/// Find-or-create `<owner>/<base>` and ensure every repo is an item. Returns the
/// collection slug.
pub fn ensure_collection(owner: &str, base: &str, repos: &[String]) -> Result<String> {
    let client = reqwest::blocking::Client::new();
    let token = hf_token();
    let auth = |rb: reqwest::blocking::RequestBuilder| match &token {
        Some(t) => rb.bearer_auth(t),
        None => rb,
    };

    let slug = match find_collection(&client, &auth, owner, base) {
        Ok(Some(slug)) => slug,
        _ => create_collection(&client, &auth, owner, base)?,
    };

    for repo in repos {
        let url = format!("{}/api/collections/{}/items", endpoint(), slug);
        let body = json!({"item": {"type": "dataset", "id": repo}});
        // Best-effort: a 409 (already an item) or any error is non-fatal.
        let _ = auth(client.post(&url)).json(&body).send();
    }
    Ok(slug)
}

fn find_collection(
    client: &reqwest::blocking::Client,
    auth: &impl Fn(reqwest::blocking::RequestBuilder) -> reqwest::blocking::RequestBuilder,
    owner: &str,
    base: &str,
) -> Result<Option<String>> {
    let url = format!("{}/api/collections", endpoint());
    let resp = auth(client.get(&url))
        .query(&[("owner", owner), ("q", base), ("limit", "20")])
        .send()?
        .error_for_status()?;
    let items: Vec<Value> = resp.json().context("parsing collections list")?;
    Ok(items
        .into_iter()
        .find(|c| c.get("title").and_then(Value::as_str) == Some(base))
        .and_then(|c| c.get("slug").and_then(Value::as_str).map(str::to_string)))
}

fn create_collection(
    client: &reqwest::blocking::Client,
    auth: &impl Fn(reqwest::blocking::RequestBuilder) -> reqwest::blocking::RequestBuilder,
    owner: &str,
    base: &str,
) -> Result<String> {
    let url = format!("{}/api/collections", endpoint());
    let body = json!({
        "title": base,
        "namespace": owner,
        "private": true,
        "description": "agentcap: paired HTTP captures + native session traces. Join on run_id.",
    });
    let resp = auth(client.post(&url)).json(&body).send()?.error_for_status()?;
    let created: Value = resp.json().context("parsing created collection")?;
    created
        .get("slug")
        .and_then(Value::as_str)
        .map(str::to_string)
        .context("created collection response missing slug")
}
