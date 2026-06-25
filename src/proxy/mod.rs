//! Synchronous capture proxy. Ports `proxy.py`.
//!
//! A `tiny_http` server on a worker-thread pool fronts an OpenAI-compatible
//! upstream. `POST /v1/chat/completions` is captured to
//! `<capture_dir>/<rid>.{request,response}.json`; every other path passes
//! through. Streaming responses are forwarded chunk-by-chunk while a tee
//! accumulates the raw bytes, persisted once the stream ends.
//!
//! Each request is handled in isolation: its own rid, its own immutable
//! `(task_id, turn)` snapshot taken at arrival, and its own response
//! accumulator — captures never share mutable state. Turns run serially, so the
//! snapshot is always the right turn.

mod capture;

use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use anyhow::{Context, Result};
use reqwest::blocking::Client;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use serde_json::Value;
use tiny_http::{Header, Method, Request, Response, Server, StatusCode};

const CHAT_COMPLETIONS_PATH: &str = "/v1/chat/completions";
const DEFAULT_WORKERS: usize = 4;

/// Hop-by-hop headers (RFC 7230 §6.1) + content-length/-encoding, which the
/// framework recomputes from the re-emitted body. Compared case-insensitively.
const HOP_BY_HOP: &[&str] = &[
    "host",
    "content-length",
    "content-encoding",
    "transfer-encoding",
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "upgrade",
];

fn is_hop_by_hop(name: &str) -> bool {
    HOP_BY_HOP.iter().any(|h| name.eq_ignore_ascii_case(h))
}

fn now_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

struct CaptureProxy {
    upstream: String,
    capture_dir: PathBuf,
    client: Client,
    ctx: Mutex<(Option<String>, Option<i64>)>,
}

impl CaptureProxy {
    fn new(upstream: &str, capture_dir: PathBuf) -> Result<Self> {
        std::fs::create_dir_all(&capture_dir).with_context(|| format!("creating {}", capture_dir.display()))?;
        // Generous per-request cap: blocking's 30s default truncates slow streamed
        // generations; finite so a hung upstream can't wedge a worker forever.
        let client = Client::builder()
            .timeout(Duration::from_secs(900))
            .build()
            .context("building HTTP client")?;
        Ok(CaptureProxy {
            upstream: upstream.trim_end_matches('/').to_string(),
            capture_dir,
            client,
            ctx: Mutex::new((None, None)),
        })
    }

    fn set_context(&self, task_id: Option<&str>, turn: Option<i64>) {
        *self.ctx.lock().unwrap() = (task_id.map(str::to_string), turn);
    }

    fn snapshot(&self) -> (Option<String>, Option<i64>) {
        self.ctx.lock().unwrap().clone()
    }

    fn handle(&self, req: Request) {
        let path = req.url().split('?').next().unwrap_or("").to_string();
        let is_chat = *req.method() == Method::Post && path == CHAT_COMPLETIONS_PATH;
        let result = if is_chat {
            self.chat_completions(req)
        } else {
            self.passthrough(req)
        };
        if let Err(e) = result {
            eprintln!("  [proxy] {e:#}");
        }
    }

    fn chat_completions(&self, mut req: Request) -> Result<()> {
        let mut body = Vec::new();
        req.as_reader().read_to_end(&mut body)?;
        let is_stream = serde_json::from_slice::<Value>(&body)
            .ok()
            .and_then(|v| v.get("stream").and_then(Value::as_bool))
            .unwrap_or(false);

        let rid = uuid::Uuid::new_v4().simple().to_string();
        let (task_id, turn) = self.snapshot();
        let _ = capture::write_request(
            &self.capture_dir,
            &rid,
            &body,
            &self.upstream,
            task_id.as_deref(),
            turn,
            now_secs(),
        );

        let url = format!("{}{}", self.upstream, CHAT_COMPLETIONS_PATH);
        let fwd = forward_headers(req.headers());
        if is_stream {
            self.forward_stream(req, &url, body, fwd, &rid)
        } else {
            self.forward_nonstream(req, &url, body, fwd, &rid)
        }
    }

    fn forward_nonstream(&self, req: Request, url: &str, body: Vec<u8>, fwd: HeaderMap, rid: &str) -> Result<()> {
        let up = match self.client.post(url).headers(fwd).body(body).send() {
            Ok(up) => up,
            Err(e) => return respond_502(req, &e),
        };
        let status = up.status().as_u16();
        let up_headers = up.headers().clone();
        let resp_bytes = up.bytes().context("reading upstream response")?.to_vec();
        let _ = capture::write_response_nonstream(&self.capture_dir, rid, status, &resp_bytes, now_secs(), &up_headers);

        let mut resp = Response::from_data(resp_bytes).with_status_code(StatusCode(status));
        for h in response_headers(&up_headers) {
            resp.add_header(h);
        }
        req.respond(resp).context("responding to agent")
    }

    fn forward_stream(&self, req: Request, url: &str, body: Vec<u8>, fwd: HeaderMap, rid: &str) -> Result<()> {
        let up = match self.client.post(url).headers(fwd).body(body).send() {
            Ok(up) => up,
            Err(e) => return respond_502(req, &e),
        };
        let status = up.status().as_u16();
        let up_headers = up.headers().clone();

        let buf = Arc::new(Mutex::new(Vec::<u8>::new()));
        let tee = TeeReader {
            inner: up,
            buf: buf.clone(),
        };
        let resp = Response::new(StatusCode(status), response_headers(&up_headers), tee, None, None);
        // respond() streams to the agent (filling `buf`) and returns at EOF or on
        // a mid-stream client disconnect; persist whatever we accumulated either way.
        let respond_result = req.respond(resp);
        let raw = buf.lock().unwrap();
        let _ = capture::write_response_stream(&self.capture_dir, rid, status, &raw, now_secs(), &up_headers);
        respond_result.context("streaming to agent")
    }

    fn passthrough(&self, mut req: Request) -> Result<()> {
        let url = format!("{}{}", self.upstream, req.url());
        let method = reqwest::Method::from_bytes(req.method().to_string().as_bytes()).unwrap_or(reqwest::Method::GET);
        let fwd = forward_headers(req.headers());
        let mut body = Vec::new();
        req.as_reader().read_to_end(&mut body)?;
        let mut builder = self.client.request(method, &url).headers(fwd);
        if !body.is_empty() {
            builder = builder.body(body);
        }
        let up = match builder.send() {
            Ok(up) => up,
            Err(e) => return respond_502(req, &e),
        };
        let status = up.status().as_u16();
        let up_headers = up.headers().clone();
        let resp_bytes = up.bytes().context("reading upstream passthrough")?.to_vec();
        let mut resp = Response::from_data(resp_bytes).with_status_code(StatusCode(status));
        for h in response_headers(&up_headers) {
            resp.add_header(h);
        }
        req.respond(resp).context("responding to agent")
    }
}

fn respond_502(req: Request, e: &reqwest::Error) -> Result<()> {
    let _ = req.respond(Response::from_string(format!("agentcap proxy: upstream error: {e}")).with_status_code(502));
    Err(anyhow::anyhow!("upstream error: {e}"))
}

/// Build a reqwest HeaderMap from the agent's request headers, dropping hop-by-hop.
fn forward_headers(headers: &[Header]) -> HeaderMap {
    let mut map = HeaderMap::new();
    for h in headers {
        let name = h.field.as_str().as_str();
        if is_hop_by_hop(name) {
            continue;
        }
        if let (Ok(n), Ok(v)) = (
            HeaderName::from_bytes(name.as_bytes()),
            HeaderValue::from_bytes(h.value.as_str().as_bytes()),
        ) {
            map.insert(n, v);
        }
    }
    map
}

/// Build tiny_http response headers from the upstream's, dropping hop-by-hop.
fn response_headers(headers: &HeaderMap) -> Vec<Header> {
    headers
        .iter()
        .filter(|(name, _)| !is_hop_by_hop(name.as_str()))
        .filter_map(|(name, value)| Header::from_bytes(name.as_str().as_bytes(), value.as_bytes()).ok())
        .collect()
}

/// Reads from `inner` while appending every byte to `buf` — streams the SSE
/// response to the agent while accumulating the raw bytes for the capture.
struct TeeReader<R: Read> {
    inner: R,
    buf: Arc<Mutex<Vec<u8>>>,
}

impl<R: Read> Read for TeeReader<R> {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        let n = self.inner.read(out)?;
        if n > 0 {
            self.buf.lock().unwrap().extend_from_slice(&out[..n]);
        }
        Ok(n)
    }
}

/// A running in-process proxy on a worker-thread pool.
pub struct ProxyHandle {
    pub port: u16,
    proxy: Arc<CaptureProxy>,
    running: Arc<AtomicBool>,
    threads: Vec<JoinHandle<()>>,
}

impl ProxyHandle {
    /// Stamp `(task_id, turn)` onto subsequent captures.
    pub fn set_context(&self, task_id: Option<&str>, turn: Option<i64>) {
        self.proxy.set_context(task_id, turn);
    }

    /// URL the sandboxed agent dials (the container reaches the host via
    /// `host.containers.internal`).
    pub fn proxy_url(&self) -> String {
        format!("http://host.containers.internal:{}/v1", self.port)
    }

    pub fn shutdown(self) {
        self.running.store(false, Ordering::Relaxed);
        // Workers poll the flag via recv_timeout, so they exit within one tick.
        for t in self.threads {
            let _ = t.join();
        }
    }
}

/// Start the proxy on a worker-thread pool, bound to `host:0` (kernel-assigned
/// port, read back into `ProxyHandle.port`). `run` binds `0.0.0.0` so the
/// container can dial in via `host.containers.internal`.
pub fn serve_in_thread(upstream: &str, capture_dir: &Path, host: &str) -> Result<ProxyHandle> {
    let proxy = Arc::new(CaptureProxy::new(upstream, capture_dir.to_path_buf())?);
    let server = Server::http((host, 0u16)).map_err(|e| anyhow::anyhow!("binding proxy: {e}"))?;
    let port = server
        .server_addr()
        .to_ip()
        .map(|a| a.port())
        .context("proxy bound to a non-IP address")?;
    let server = Arc::new(server);
    let running = Arc::new(AtomicBool::new(true));

    let mut threads = Vec::with_capacity(DEFAULT_WORKERS);
    for _ in 0..DEFAULT_WORKERS {
        let (server, proxy, running) = (server.clone(), proxy.clone(), running.clone());
        threads.push(std::thread::spawn(move || {
            while running.load(Ordering::Relaxed) {
                match server.recv_timeout(Duration::from_millis(250)) {
                    Ok(Some(req)) => proxy.handle(req),
                    Ok(None) => continue, // tick: re-check the running flag
                    Err(_) => break,      // fatal
                }
            }
        }));
    }
    Ok(ProxyHandle {
        port,
        proxy,
        running,
        threads,
    })
}
