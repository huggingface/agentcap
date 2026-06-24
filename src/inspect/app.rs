//! The inspect TUI: a stack of picker levels (run → request → message, or
//! parquet/hf variants) with synchronous nucleo filtering and a live preview
//! pane. Esc pops a level (exits past the root); Enter drills in.

use std::io::{self, Stdout};
use std::sync::mpsc::Receiver;
use std::sync::Arc;

use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen};
use nucleo::pattern::{CaseMatching, Normalization, Pattern};
use nucleo::{Config, Matcher, Utf32Str};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Modifier, Style};
use ratatui::text::Line;
use ratatui::widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Wrap};
use ratatui::{Frame, Terminal};

use crate::diff::flatten;
use crate::hub::footer::FooterMeta;
use crate::hub::DatasetHandle;
use crate::query::parse_fzf_terms;

use super::render;
use super::sources::{
    enumerate_workspace_requests, parquet_request_level, request_messages_for_view, MsgRecord, ReqRow, ReqSource,
    RunRow,
};

const POLL_MS: u64 = 100;
const PAGE: usize = 10;

/// An HF parquet file row whose metadata streams in from background workers.
pub struct HfRow {
    pub path: String,
    pub agent: Option<String>,
    pub model: Option<String>,
    pub num_rows: Option<i64>,
    pub tasks: Option<Vec<(String, i64, Option<String>)>>,
}

enum LevelKind {
    Run(Vec<RunRow>),
    Request {
        rows: Vec<ReqRow>,
        source: ReqSource,
    },
    Message(Vec<MsgRecord>),
    Hf {
        repo: DatasetHandle,
        rows: Vec<HfRow>,
        rx: Receiver<(usize, FooterMeta)>,
    },
}

struct DisplayRow {
    text: String,
    header: bool,
    haystack: String,
}

pub struct Level {
    kind: LevelKind,
    display: Vec<DisplayRow>,
    query: String,
    selected: usize,
    filtered: Vec<usize>,
}

impl Level {
    fn run(rows: Vec<RunRow>) -> Self {
        let display = build_run_display(&rows);
        Self::wrap(LevelKind::Run(rows), display)
    }

    fn request(rows: Vec<ReqRow>, source: ReqSource) -> Self {
        let include_run = rows
            .iter()
            .map(|r| r.run_id.as_str())
            .collect::<std::collections::HashSet<_>>()
            .len()
            > 1;
        let loc_w = "LOC"
            .len()
            .max(rows.iter().map(|r| loc_cell(r).len()).max().unwrap_or(0));
        let run_w = if include_run {
            "RUN".len().max(rows.iter().map(|r| r.run_id.len()).max().unwrap_or(0))
        } else {
            0
        };
        let display = build_request_display(&rows, include_run, loc_w, run_w);
        Self::wrap(LevelKind::Request { rows, source }, display)
    }

    fn message(records: Vec<MsgRecord>) -> Self {
        let display = build_message_display(&records);
        Self::wrap(LevelKind::Message(records), display)
    }

    fn hf(repo: DatasetHandle, rows: Vec<HfRow>, rx: Receiver<(usize, FooterMeta)>) -> Self {
        let display = build_hf_display(&rows);
        Self::wrap(LevelKind::Hf { repo, rows, rx }, display)
    }

    fn wrap(kind: LevelKind, display: Vec<DisplayRow>) -> Self {
        let filtered = (0..display.len()).collect();
        Level {
            kind,
            display,
            query: String::new(),
            selected: 0,
            filtered,
        }
    }

    fn title(&self) -> &'static str {
        match self.kind {
            LevelKind::Run(_) => "runs",
            LevelKind::Request { .. } => "requests",
            LevelKind::Message(_) => "messages",
            LevelKind::Hf { .. } => "parquet files",
        }
    }

    fn refilter(&mut self, matcher: &mut Matcher, buf: &mut Vec<char>) {
        if self.query.is_empty() {
            self.filtered = (0..self.display.len()).collect();
        } else {
            let pattern = Pattern::parse(&self.query, CaseMatching::Smart, Normalization::Smart);
            let mut scored: Vec<(usize, u32)> = self
                .display
                .iter()
                .enumerate()
                .filter_map(|(i, d)| {
                    let h = Utf32Str::new(&d.haystack, buf);
                    pattern.score(h, matcher).map(|s| (i, s))
                })
                .collect();
            scored.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
            self.filtered = scored.into_iter().map(|(i, _)| i).collect();
        }
        if self.selected >= self.filtered.len() {
            self.selected = self.filtered.len().saturating_sub(1);
        }
    }

    fn selected_row(&self) -> Option<usize> {
        self.filtered.get(self.selected).copied()
    }

    /// Drain newly-arrived HF footer metadata; returns true if anything changed.
    fn drain_hf(&mut self) -> bool {
        let mut changed = false;
        if let LevelKind::Hf { rows, rx, .. } = &mut self.kind {
            while let Ok((idx, meta)) = rx.try_recv() {
                if let Some(row) = rows.get_mut(idx) {
                    row.agent = meta.agent;
                    row.model = meta.model;
                    row.num_rows = Some(meta.num_rows);
                    row.tasks = meta.tasks;
                    changed = true;
                }
            }
            if changed {
                self.display = build_hf_display(rows);
            }
        }
        changed
    }

    fn preview(&self, terms: &[String]) -> ratatui::text::Text<'static> {
        let Some(i) = self.selected_row() else {
            return ratatui::text::Text::raw("");
        };
        match &self.kind {
            LevelKind::Run(rows) => render::run_preview(&rows[i]),
            LevelKind::Message(recs) => render::message_preview(&recs[i], terms),
            LevelKind::Hf { rows, .. } => {
                let r = &rows[i];
                render::hf_parquet_preview(&r.path, &r.agent, &r.model, r.num_rows, &r.tasks)
            }
            LevelKind::Request { rows, source, .. } => {
                let row = &rows[i];
                let body = source.load_body(&row.rid);
                let prev_body = row.prev_rid.as_ref().map(|p| source.load_body(p));
                render::request_preview(row, &body, prev_body.as_ref(), terms)
            }
        }
    }

    /// Drill into the selected row; `Ok(None)` for a leaf or no selection.
    fn enter(&self) -> Result<Option<Level>> {
        let Some(i) = self.selected_row() else { return Ok(None) };
        match &self.kind {
            LevelKind::Run(rows) => {
                let reqs = enumerate_workspace_requests(&rows[i].run_dir);
                let source = ReqSource::Workspace {
                    cap_dir: rows[i].run_dir.join("captures"),
                };
                Ok(Some(Level::request(reqs, source)))
            }
            LevelKind::Request { rows, source, .. } => {
                let row = &rows[i];
                let body = source.load_body(&row.rid);
                let resp = source.load_resp(&row.rid);
                let records = request_messages_for_view(&body, resp.as_ref());
                Ok(Some(Level::message(records)))
            }
            LevelKind::Message(_) => Ok(None),
            LevelKind::Hf { repo, rows, .. } => {
                let local = crate::hub::download_file(repo, &rows[i].path)?;
                let (reqs, map) = parquet_request_level(&local)?;
                Ok(Some(Level::request(reqs, ReqSource::Parquet { rows: Arc::new(map) })))
            }
        }
    }
}

fn loc_cell(r: &ReqRow) -> String {
    match (&r.task_id, r.req_index) {
        (Some(t), idx) if !t.is_empty() => format!("{t}.{idx}"),
        _ => "-".to_string(),
    }
}

fn build_run_display(rows: &[RunRow]) -> Vec<DisplayRow> {
    let aw = "AGENT".len().max(rows.iter().map(|r| r.agent.len()).max().unwrap_or(0));
    let mw = "MODEL".len().max(rows.iter().map(|r| r.model.len()).max().unwrap_or(0));
    rows.iter()
        .map(|r| DisplayRow {
            text: format!(
                "{:<aw$}  {:<mw$}  {:>5}  {:>5}",
                r.agent,
                r.model,
                r.n_tasks,
                r.n_caps,
                aw = aw,
                mw = mw
            ),
            header: false,
            haystack: format!("{} {} {}", r.agent, r.model, r.run_id),
        })
        .collect()
}

fn build_request_display(rows: &[ReqRow], include_run: bool, loc_w: usize, run_w: usize) -> Vec<DisplayRow> {
    let mut out = Vec::with_capacity(rows.len());
    let mut prev_task: Option<String> = None;
    for r in rows {
        let loc = loc_cell(r);
        let rid8: String = r.rid.chars().take(8).collect();
        let mut text = format!("{loc:<loc_w$}  {rid8:<8}");
        if include_run {
            text.push_str(&format!("  {:<run_w$}", r.run_id, run_w = run_w));
        }
        text.push_str(&format!("  {}", r.preview));
        let text = text.replace('\t', " ");
        let header = r
            .task_id
            .as_ref()
            .is_some_and(|t| !t.is_empty() && Some(t) != prev_task.as_ref());
        prev_task = r.task_id.clone();
        let haystack = format!("{text} {} {}", r.rid, r.searchable);
        out.push(DisplayRow { text, header, haystack });
    }
    out
}

fn build_message_display(recs: &[MsgRecord]) -> Vec<DisplayRow> {
    recs.iter()
        .enumerate()
        .map(|(i, m)| DisplayRow {
            text: format!("[{:>3}] {:<14} {}", i + 1, m.role, flatten(&m.summary, 200)),
            header: false,
            haystack: format!("{} {}", m.role, m.summary),
        })
        .collect()
}

fn build_hf_display(rows: &[HfRow]) -> Vec<DisplayRow> {
    let q = |o: &Option<String>| o.clone().unwrap_or_else(|| "?".to_string());
    let aw = "AGENT"
        .len()
        .max(rows.iter().map(|r| q(&r.agent).len()).max().unwrap_or(0));
    let mw = "MODEL"
        .len()
        .max(rows.iter().map(|r| q(&r.model).len()).max().unwrap_or(0));
    rows.iter()
        .map(|r| {
            let tasks = r
                .tasks
                .as_ref()
                .map(|t| t.len().to_string())
                .unwrap_or_else(|| "?".to_string());
            let caps = r.num_rows.map(|n| n.to_string()).unwrap_or_else(|| "?".to_string());
            DisplayRow {
                text: format!(
                    "{:<aw$}  {:<mw$}  {:>5}  {:>6}",
                    q(&r.agent),
                    q(&r.model),
                    tasks,
                    caps,
                    aw = aw,
                    mw = mw
                ),
                header: false,
                haystack: format!("{} {} {}", q(&r.agent), q(&r.model), r.path),
            }
        })
        .collect()
}

/// RAII terminal guard — restores cooked mode + main screen on drop.
struct Tui {
    term: Terminal<CrosstermBackend<Stdout>>,
}

impl Tui {
    fn new() -> Result<Self> {
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen)?;
        let term = Terminal::new(CrosstermBackend::new(stdout))?;
        Ok(Tui { term })
    }
}

impl Drop for Tui {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(self.term.backend_mut(), LeaveAlternateScreen);
        let _ = self.term.show_cursor();
    }
}

/// Run the picker over an initial stack (its last element is the active level).
pub fn run(initial: Vec<Level>) -> Result<()> {
    let mut stack = initial;
    if stack.is_empty() {
        return Ok(());
    }
    let mut matcher = Matcher::new(Config::DEFAULT);
    let mut buf: Vec<char> = Vec::new();
    let mut tui = Tui::new()?;
    // Preview is expensive (loads bodies); rebuild it only when the active level,
    // selection, query, or streamed HF metadata changes — not every poll tick.
    let mut cached: Option<(PreviewKey, ratatui::text::Text<'static>)> = None;
    let mut hf_version: u64 = 0;

    loop {
        let key = {
            let top = stack.last().unwrap();
            PreviewKey {
                depth: stack.len(),
                selected: top.selected,
                query: top.query.clone(),
                hf_version,
            }
        };
        if cached.as_ref().map(|(k, _)| k != &key).unwrap_or(true) {
            let top = stack.last().unwrap();
            let terms = parse_fzf_terms(&top.query);
            let preview = top.preview(&terms);
            cached = Some((key, preview));
        }
        {
            let top = stack.last().unwrap();
            let preview = &cached.as_ref().unwrap().1;
            tui.term.draw(|f| draw(f, &stack, top, preview))?;
        }

        // Keep HF rows flowing in even when the user isn't typing.
        if event::poll(std::time::Duration::from_millis(POLL_MS))? {
            if let Event::Key(key) = event::read()? {
                if key.kind != KeyEventKind::Press {
                    continue;
                }
                if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
                    break;
                }
                match key.code {
                    KeyCode::Esc => {
                        stack.pop();
                        if stack.is_empty() {
                            break;
                        }
                    }
                    KeyCode::Enter => {
                        if let Some(next) = stack.last().unwrap().enter()? {
                            stack.push(next);
                        }
                    }
                    KeyCode::Up => move_sel(stack.last_mut().unwrap(), -1),
                    KeyCode::Down => move_sel(stack.last_mut().unwrap(), 1),
                    KeyCode::PageUp => move_sel(stack.last_mut().unwrap(), -(PAGE as isize)),
                    KeyCode::PageDown => move_sel(stack.last_mut().unwrap(), PAGE as isize),
                    KeyCode::Home => stack.last_mut().unwrap().selected = 0,
                    KeyCode::End => {
                        let top = stack.last_mut().unwrap();
                        top.selected = top.filtered.len().saturating_sub(1);
                    }
                    KeyCode::Backspace => {
                        let top = stack.last_mut().unwrap();
                        top.query.pop();
                        top.refilter(&mut matcher, &mut buf);
                    }
                    KeyCode::Char(c) => {
                        let top = stack.last_mut().unwrap();
                        top.query.push(c);
                        top.refilter(&mut matcher, &mut buf);
                    }
                    _ => {}
                }
            }
        }

        // Drain any HF metadata that arrived; refilter so new haystacks apply,
        // and bump the version so the preview cache invalidates.
        if stack.last_mut().unwrap().drain_hf() {
            let top = stack.last_mut().unwrap();
            top.refilter(&mut matcher, &mut buf);
            hf_version += 1;
        }
    }
    Ok(())
}

/// Identity of the currently-previewed row; the preview is recomputed only when
/// this changes.
#[derive(PartialEq)]
struct PreviewKey {
    depth: usize,
    selected: usize,
    query: String,
    hf_version: u64,
}

fn move_sel(level: &mut Level, delta: isize) {
    let n = level.filtered.len();
    if n == 0 {
        return;
    }
    let cur = level.selected as isize;
    level.selected = cur.saturating_add(delta).clamp(0, n as isize - 1) as usize;
}

fn draw(f: &mut Frame, stack: &[Level], top: &Level, preview: &ratatui::text::Text<'static>) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Min(1), Constraint::Length(1)])
        .split(f.area());

    let depth = stack.len();
    let header = format!(
        " agentcap inspect · {} (level {depth}) — Enter: drill · Esc: back · type to filter ",
        top.title()
    );
    f.render_widget(
        Paragraph::new(header).style(Style::default().add_modifier(Modifier::REVERSED)),
        chunks[0],
    );

    let body = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(45), Constraint::Percentage(55)])
        .split(chunks[1]);

    let items: Vec<ListItem> = top
        .filtered
        .iter()
        .map(|&i| {
            let d = &top.display[i];
            let style = if d.header {
                Style::default().add_modifier(Modifier::REVERSED)
            } else {
                Style::default()
            };
            ListItem::new(Line::styled(d.text.clone(), style))
        })
        .collect();
    let list = List::new(items)
        .block(Block::default().borders(Borders::RIGHT))
        .highlight_style(Style::default().add_modifier(Modifier::REVERSED | Modifier::BOLD))
        .highlight_symbol("▶ ");
    let mut state = ListState::default();
    if !top.filtered.is_empty() {
        state.select(Some(top.selected));
    }
    f.render_stateful_widget(list, body[0], &mut state);

    f.render_widget(Paragraph::new(preview.clone()).wrap(Wrap { trim: false }), body[1]);

    let input = format!("> {}   [{} / {}]", top.query, top.filtered.len(), top.display.len());
    f.render_widget(Paragraph::new(input), chunks[2]);
}

// --- constructors used by `mod.rs` to seed the initial stack ---

pub fn level_run(rows: Vec<RunRow>) -> Level {
    Level::run(rows)
}

pub fn level_request_workspace(run_dir: &std::path::Path) -> Level {
    let reqs = enumerate_workspace_requests(run_dir);
    Level::request(
        reqs,
        ReqSource::Workspace {
            cap_dir: run_dir.join("captures"),
        },
    )
}

pub fn level_request_parquet(path: std::path::PathBuf) -> Result<Level> {
    let (reqs, map) = parquet_request_level(&path)?;
    Ok(Level::request(reqs, ReqSource::Parquet { rows: Arc::new(map) }))
}

pub fn level_hf(repo: DatasetHandle, rows: Vec<HfRow>, rx: Receiver<(usize, FooterMeta)>) -> Level {
    Level::hf(repo, rows, rx)
}
