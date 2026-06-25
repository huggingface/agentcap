//! Read a parquet file's metadata (schema KV + row count) over `hf://` via a
//! ranged footer read — no full download. Powers the inspect HF picker.
//!
//! Our writer stamps `agent`/`model`/`tasks` into the parquet key-value
//! metadata, so a footer read surfaces the full preview slice. (Parquets written
//! by pyarrow keep those under the embedded `ARROW:schema` blob instead; those
//! show only the row count here until selected.)

use anyhow::{bail, Result};
use parquet::errors::ParquetError;
use parquet::file::metadata::ParquetMetaDataReader;
use serde_json::Value;

use super::DatasetHandle;

#[derive(Debug, Clone, Default)]
pub struct FooterMeta {
    pub agent: Option<String>,
    pub model: Option<String>,
    pub num_rows: i64,
    /// `(id, turns, prompt)` per task, when the `tasks` KV is present.
    pub tasks: Option<Vec<(String, i64, Option<String>)>>,
}

const INITIAL_SUFFIX: u64 = 64 * 1024;

/// Decode parquet metadata for `path` (size `size`) using ranged reads. Starts
/// with a 64 KiB suffix and grows if the footer needs more (the standard
/// `try_parse_sized` negotiation).
pub fn fetch_parquet_meta(repo: &DatasetHandle, path: &str, size: u64) -> Result<FooterMeta> {
    if size < 8 {
        bail!("{path}: too small to be a parquet ({size} bytes)");
    }
    let mut want = INITIAL_SUFFIX.min(size);
    loop {
        let bytes = super::download_range(repo, path, (size - want)..size)?;
        let mut reader = ParquetMetaDataReader::new();
        match reader.try_parse_sized(&bytes, size) {
            Ok(()) => return Ok(extract(&reader.finish()?)),
            Err(ParquetError::NeedMoreData(needed)) => {
                let needed = needed as u64;
                if needed > size || needed <= want {
                    bail!("{path}: parquet footer read did not converge");
                }
                want = needed.min(size);
            }
            Err(e) => bail!("{path}: {e}"),
        }
    }
}

fn extract(md: &parquet::file::metadata::ParquetMetaData) -> FooterMeta {
    let fm = md.file_metadata();
    let mut out = FooterMeta {
        num_rows: fm.num_rows(),
        ..Default::default()
    };
    if let Some(kvs) = fm.key_value_metadata() {
        for kv in kvs {
            let Some(v) = &kv.value else { continue };
            match kv.key.as_str() {
                "agent" => out.agent = Some(v.clone()),
                "model" => out.model = Some(v.clone()),
                "tasks" => {
                    if let Ok(Value::Array(arr)) = serde_json::from_str::<Value>(v) {
                        out.tasks = Some(
                            arr.iter()
                                .map(|t| {
                                    (
                                        t.get("id").and_then(Value::as_str).unwrap_or("?").to_string(),
                                        t.get("turns").and_then(Value::as_i64).unwrap_or(0),
                                        t.get("prompt").and_then(Value::as_str).map(str::to_string),
                                    )
                                })
                                .collect(),
                        );
                    }
                }
                _ => {}
            }
        }
    }
    out
}
