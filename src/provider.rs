//! Identify the inference backend behind an upstream URL.
//!
//! Hostname classification ([`hostname_fallback`]) + the HF Router sub-provider pin
//! ([`refine_for_sub_provider`]) — what `run` and `export` use to slug a backend.
//! Live network introspection of the backend isn't implemented; the hostname slug
//! is enough for both paths.

use std::net::IpAddr;

/// Reverse proxies / custom domains won't match; the probe path catches those.
const HOSTNAME_TO_PROVIDER: &[(&str, &str)] = &[
    ("router.huggingface.co", "hf-router"),
    ("api.openai.com", "openai"),
    ("api.together.xyz", "together"),
    ("api.anthropic.com", "anthropic"),
    ("api.cerebras.ai", "cerebras"),
    ("api.fireworks.ai", "fireworks"),
    ("api.groq.com", "groq"),
];

/// Derive a provider slug from the upstream URL's hostname: known hosts map
/// directly; loopback / RFC1918 → `local`; otherwise the eTLD+1-ish second-level
/// label (`api.mycompany.com` → `mycompany`).
pub fn hostname_fallback(upstream_url: &str) -> String {
    let host = match url::Url::parse(upstream_url) {
        Ok(u) => u.host_str().unwrap_or("").to_lowercase(),
        Err(_) => String::new(),
    };
    // url keeps IPv6 hosts bracketed (`[::1]`); strip for the literal match below.
    let host = host.trim_start_matches('[').trim_end_matches(']').to_string();
    if host.is_empty() {
        return "unknown".to_string();
    }
    if let Some((_, p)) = HOSTNAME_TO_PROVIDER.iter().find(|(h, _)| *h == host) {
        return p.to_string();
    }
    if host == "localhost" || host == "::1" {
        return "local".to_string();
    }
    if let Ok(ip) = host.parse::<IpAddr>() {
        let is_local = ip.is_loopback() || matches!(ip, IpAddr::V4(v4) if v4.is_private());
        return if is_local { "local".to_string() } else { host };
    }
    let parts: Vec<&str> = host.split('.').collect();
    if parts.len() >= 2 {
        parts[parts.len() - 2].to_string()
    } else {
        host
    }
}

/// Surface HF Router's `meta-llama/...:fireworks-ai` pin as
/// `hf-router/fireworks-ai` in the provider slug.
pub fn refine_for_sub_provider(provider: &str, model: Option<&str>) -> String {
    if provider == "hf-router" {
        if let Some(m) = model {
            if let Some((_, sub)) = m.split_once(':') {
                return format!("hf-router/{sub}");
            }
        }
    }
    provider.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_providers() {
        assert_eq!(hostname_fallback("https://router.huggingface.co/v1"), "hf-router");
        assert_eq!(hostname_fallback("https://api.openai.com/v1"), "openai");
        assert_eq!(hostname_fallback("https://api.together.xyz/v1"), "together");
        assert_eq!(hostname_fallback("https://api.fireworks.ai/v1"), "fireworks");
    }

    #[test]
    fn loopback_and_private() {
        assert_eq!(hostname_fallback("http://127.0.0.1:8000/v1"), "local");
        assert_eq!(hostname_fallback("http://localhost:8000/v1"), "local");
        assert_eq!(hostname_fallback("http://10.0.0.5:8000/v1"), "local");
        assert_eq!(hostname_fallback("http://192.168.1.42:8000/v1"), "local");
    }

    #[test]
    fn unknown_public() {
        assert_eq!(hostname_fallback("https://api.mycompany.com/v1"), "mycompany");
    }

    #[test]
    fn refine_pins_and_noops() {
        assert_eq!(
            refine_for_sub_provider("hf-router", Some("meta-llama/Llama-3.3-70B-Instruct:fireworks-ai")),
            "hf-router/fireworks-ai"
        );
        assert_eq!(
            refine_for_sub_provider("hf-router", Some("meta-llama/Llama-3.3-70B")),
            "hf-router"
        );
        assert_eq!(refine_for_sub_provider("local", Some("anything:fireworks-ai")), "local");
    }
}
