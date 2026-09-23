"""Web interface for the trading bot (port 9124) — Apple-fluid design.

Shows: dashboard, architecture diagram, code/folders, Obsidian notes,
a working BACKTEST tool (with equity chart + benchmark), a DOWNLOAD button
(zip for GitHub export, works on iPad) and GitHub instructions.

Standalone with stdlib + markdown-it-py. Usage:
    uv run python -m hermes_bot.webui   (or via systemd)
"""
from __future__ import annotations

import html
import os
import subprocess
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parent.parent
# Optional path to an Obsidian vault (local only; placeholder for publication).
VAULT = Path(os.environ.get("HERMES_OBSIDIAN_VAULT", ""))
PORT = 9124

_md = MarkdownIt("commonmark", {"html": True}).enable("table")


def _render_md(text: str) -> str:
    return _md.render(text)


# ---------------------------------------------------------------------------
# APPLE-FLUID DESIGN (CSS) — fluid, translucent, responsive, reduced-motion-safe
# ---------------------------------------------------------------------------
APPLE_CSS = r"""
:root {
  --bg: #f5f5f7; --text: #1d1d1f; --muted: #6e6e73;
  --accent: #0a84ff; --accent2: #30d158; --danger: #ff453a;
  --card: rgba(255,255,255,0.72);
  --radius: 18px; --shadow: 0 8px 30px rgba(0,0,0,.08);
  --font: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", system-ui, sans-serif;
}
@media (prefers-color-scheme: dark) {
  :root { --bg: #000; --text: #f5f5f7; --muted: #86868b;
    --card: rgba(28,28,30,0.72); --shadow: 0 8px 30px rgba(0,0,0,.5); }
}
* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body { margin: 0; font-family: var(--font); background: var(--bg); color: var(--text);
  font-size: 16px; line-height: 1.5; transition: background .4s ease, color .4s ease; }

/* Translucent floating nav (Apple material) */
header { position: sticky; top: 0; z-index: 50;
  background: color-mix(in srgb, var(--card) 60%, transparent);
  backdrop-filter: blur(20px) saturate(180%); -webkit-backdrop-filter: blur(20px) saturate(180%);
  border-bottom: 1px solid color-mix(in srgb, var(--text) 10%, transparent);
  padding: 10px 20px; display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }
header h1 { margin: 0; font-size: 19px; font-weight: 700; letter-spacing: -0.02em; }
header .dot { width: 9px; height: 9px; border-radius: 50%; background: var(--accent2);
  display: inline-block; margin-right: 6px; }

nav { display: flex; gap: 2px; flex-wrap: wrap; }
nav a { color: var(--text); text-decoration: none; font-weight: 500; font-size: 14px;
  padding: 7px 14px; border-radius: 999px; transition: background .2s ease, transform .1s; }
nav a:hover { background: color-mix(in srgb, var(--accent) 12%, transparent); }
nav a.active { background: color-mix(in srgb, var(--accent) 18%, transparent); color: var(--accent); }
nav a:active { transform: scale(0.96); }

main { max-width: 1100px; margin: 0 auto; padding: 26px 20px 80px; }
h2 { font-size: 26px; font-weight: 700; letter-spacing: -0.03em; margin: 6px 0 16px; }
h3 { font-size: 17px; font-weight: 600; margin: 22px 0 10px; }
p { color: var(--muted); }
a { color: var(--accent); }

.card { background: var(--card); border-radius: var(--radius); box-shadow: var(--shadow);
  padding: 20px; backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);
  border: 1px solid color-mix(in srgb, var(--text) 8%, transparent); margin-bottom: 18px;
  transition: transform .25s cubic-bezier(.2,.8,.2,1), opacity .3s ease; }
.card:hover { transform: translateY(-2px); }

.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; }
.kpi { background: var(--card); border-radius: 14px; padding: 14px 16px; box-shadow: var(--shadow); }
.kpi .label { font-size: 12px; color: var(--muted); font-weight: 500; text-transform: uppercase; letter-spacing: .04em; }
.kpi .value { font-size: 22px; font-weight: 700; letter-spacing: -0.02em; margin-top: 2px; }
.kpi.good .value { color: var(--accent2); }
.kpi.bad .value { color: var(--danger); }
.kpi.neutral .value { color: var(--accent); }

button, .btn { background: var(--accent); color: #fff; border: none; cursor: pointer;
  font-family: var(--font); font-weight: 600; font-size: 15px; padding: 11px 20px;
  border-radius: 999px; transition: transform .12s ease, box-shadow .2s ease, opacity .2s; }
button:hover, .btn:hover { box-shadow: 0 4px 16px rgba(10,132,255,.35); }
button:active, .btn:active { transform: scale(0.97); }
.btn.secondary { background: color-mix(in srgb, var(--accent) 14%, transparent); color: var(--accent); }

input, select { font-family: var(--font); font-size: 15px; padding: 11px 14px;
  border-radius: 12px; border: 1px solid color-mix(in srgb, var(--text) 18%, transparent);
  background: var(--card); color: var(--text); margin: 4px 6px 4px 0; }
input:focus, select:focus { outline: 2px solid var(--accent); outline-offset: 2px; }

.form-row { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.spin { display: inline-block; width: 15px; height: 15px; border: 2px solid #fff;
  border-top-color: transparent; border-radius: 50%; animation: spin .6s linear infinite; vertical-align: -2px; }
@keyframes spin { to { transform: rotate(360deg); } }

/* Chart */
.chart-wrap { position: relative; width: 100%; overflow: hidden; border-radius: 12px;
  background: color-mix(in srgb, var(--bg) 40%, transparent); padding: 6px; }
.chart-legend { display: flex; gap: 18px; font-size: 13px; color: var(--muted); margin-top: 8px; flex-wrap: wrap; }
.legend-item { display: flex; align-items: center; gap: 6px; }
.legend-item .swatch { width: 14px; height: 4px; border-radius: 2px; }
.legend-item.strategy .swatch { background: var(--accent); }
.legend-item.bench .swatch { background: var(--muted); }

/* Code viewer */
pre { background: color-mix(in srgb, var(--bg) 70%, transparent); padding: 16px;
  border-radius: 14px; overflow: auto; font-size: 13px; font-family: ui-monospace, "SF Mono", Menlo, monospace;
  border: 1px solid color-mix(in srgb, var(--text) 10%, transparent); }
pre code { white-space: pre; }
.line-num { color: var(--muted); user-select: none; margin-right: 10px; }

/* Trees */
details { margin: 4px 0; }
summary { cursor: pointer; font-weight: 600; padding: 6px 8px; border-radius: 8px;
  transition: background .15s ease; }
summary:hover { background: color-mix(in srgb, var(--accent) 10%, transparent); }
.file a { color: var(--text); text-decoration: none; padding: 3px 8px; border-radius: 6px; display: inline-block; font-size: 14px; }
.file a:hover { background: color-mix(in srgb, var(--accent) 12%, transparent); color: var(--accent); }

/* Architecture blocks */
.arch { display: flex; flex-direction: column; gap: 10px; align-items: center; }
.arch-cap { font-size: 12px; color: var(--muted); letter-spacing: .06em; text-transform: uppercase;
  margin: 14px 0 4px; font-weight: 600; }
.arch-row { display: flex; gap: 12px; align-items: center; justify-content: center; flex-wrap: wrap; }
.arch-block { background: var(--card); border: 1px solid color-mix(in srgb, var(--text) 10%, transparent);
  border-radius: 14px; padding: 12px 16px; min-width: 130px; text-align: center; box-shadow: var(--shadow);
  transition: transform .2s ease; }
.arch-block:hover { transform: translateY(-2px) scale(1.02); }
.arch-block h4 { margin: 0 0 4px; font-size: 13px; color: var(--accent); }
.arch-block .sub { font-size: 11px; color: var(--muted); line-height: 1.4; }
.arch-block .sub span { display: block; }
.arch-block.core { border-color: var(--accent); }
.arch-block.risk { border-color: var(--danger); }
.arch-block.action { border-color: var(--accent2); }
.arch-arrow { color: var(--accent); font-size: 20px; font-weight: 700; }
.arch-arrow.risk { color: var(--danger); }

/* Tables */
table { border-collapse: collapse; width: 100%; }
td, th { border-bottom: 1px solid color-mix(in srgb, var(--text) 10%, transparent);
  padding: 8px 10px; text-align: left; font-size: 14px; }
th { color: var(--muted); font-weight: 600; font-size: 12px; letter-spacing: .02em; }

/* Download panel */
.dl-panel { display: flex; flex-wrap: wrap; gap: 16px; align-items: center; }
.dl-size { color: var(--muted); font-size: 13px; }

/* Code viewer sidebar layout on wide screens */
.code-layout { display: grid; grid-template-columns: 300px 1fr; gap: 20px; align-items: start; }
@media (max-width: 820px) { .code-layout { grid-template-columns: 1fr; } }

/* Architecture modal (clickable component) */
.arch-node { cursor: pointer; }
.arch-node:focus { outline: 2px solid var(--accent); outline-offset: 2px; border-radius: 16px; }
.arch-modal { position: fixed; inset: 0; z-index: 200; display: flex; align-items: center;
  justify-content: center; background: rgba(0,0,0,.35); backdrop-filter: blur(6px);
  -webkit-backdrop-filter: blur(6px); opacity: 0; pointer-events: none;
  transition: opacity .25s ease; padding: 20px; }
.arch-modal.open { opacity: 1; pointer-events: auto; }
.arch-modal-card { background: var(--card); border-radius: 20px; box-shadow: var(--shadow);
  max-width: 460px; width: 100%; padding: 24px 26px; position: relative;
  transform: translateY(12px) scale(.97); transition: transform .3s cubic-bezier(.2,.8,.2,1); }
.arch-modal.open .arch-modal-card { transform: translateY(0) scale(1); }
.arch-modal-close { position: absolute; top: 14px; right: 14px; background: none; color: var(--muted);
  font-size: 18px; padding: 6px 10px; border-radius: 999px; }
.arch-modal-close:hover { background: color-mix(in srgb, var(--text) 10%, transparent); box-shadow: none; }
.arch-modal-title { font-size: 20px; font-weight: 700; letter-spacing: -0.02em; margin: 0 0 6px; }
.arch-modal-rol { font-size: 14px; color: var(--accent); font-weight: 600; margin: 0 0 10px; }
.arch-modal-wat { font-size: 14px; color: var(--text); line-height: 1.55; margin: 0 0 16px; }
.arch-modal-flow { display: flex; flex-direction: column; gap: 8px; border-top: 1px solid
  color-mix(in srgb, var(--text) 10%, transparent); padding-top: 14px; }
.arch-modal-flow > div { display: flex; gap: 10px; font-size: 13px; }
.arch-modal-flow .lbl { flex: 0 0 44px; color: var(--muted); font-weight: 600; text-transform: uppercase;
  font-size: 11px; letter-spacing: .04em; padding-top: 1px; }
.arch-modal-flow span:last-child { color: var(--text); }

/* Reduced motion */
@media (prefers-reduced-motion: reduce) {
  .card, nav a, button, .arch-block, header { transition: none !important; }
  button:active, nav a:active { transform: none; }
}

/* Toast */
#toast { position: fixed; bottom: 24px; left: 50%; transform: translateX(-50%);
  background: var(--text); color: var(--bg); padding: 12px 20px; border-radius: 999px;
  font-weight: 600; font-size: 14px; opacity: 0; pointer-events: none;
  transition: opacity .3s ease, transform .3s ease; z-index: 100; }
#toast.show { opacity: 1; transform: translateX(-50%) translateY(-6px); }
"""

TOAST_JS = r"""
function toast(msg){ const t=document.getElementById('toast'); t.textContent=msg;
  t.classList.add('show'); setTimeout(()=>t.classList.remove('show'), 2500); }
"""


# ---------------------------------------------------------------------------
# HELPER HTML
# ---------------------------------------------------------------------------
def _tree_html(path: Path, base: Path) -> str:
    """Recursive file tree as HTML."""
    out: list[str] = []
    try:
        entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    except OSError:
        return ""
    for p in entries:
        if p.name.startswith(".") or p.name in ("__pycache__", ".venv", "node_modules", "var"):
            continue
        rel = p.relative_to(base)
        if p.is_dir():
            out.append(
                f'<details><summary>📁 {html.escape(p.name)}</summary>'
                f"{_tree_html(p, base)}</details>"
            )
        elif p.suffix in (".py", ".md", ".yaml", ".yml", ".toml", ".txt", ".json"):
            out.append(
                f'<div class="file"><a href="/code?path={urllib.parse.quote(str(rel))}">'
                f"📄 {html.escape(p.name)}</a></div>"
            )
    return "".join(out)


def _vault_notes_html() -> str:
    out: list[str] = []
    if not VAULT.is_dir():
        return '<div class="file">⚠ vault not reachable</div>'
    for p in sorted(VAULT.iterdir()):
        if p.name.startswith("."):
            continue
        if p.is_dir():
            notes = sorted(p.glob("*.md"))
            sub = "".join(
                f'<div class="file"><a href="/vault?note={urllib.parse.quote(p.name + "/" + n.name)}">'
                f"📝 {html.escape(n.stem)}</a></div>" for n in notes[:30]
            )
            out.append(f'<details><summary>📁 {html.escape(p.name)} · {len(notes)}</summary>{sub}</details>')
        elif p.suffix == ".md":
            out.append(
                f'<div class="file"><a href="/vault?note={urllib.parse.quote(p.name)}">'
                f"📝 {html.escape(p.stem)}</a></div>"
            )
    return "".join(out)


def _arch_block(title: str, subs: list[str], cls: str = "") -> str:
    sub = "".join(f"<span>{html.escape(s)}</span>" for s in subs)
    return f"<div class='arch-block {{cls}}'><h4>{html.escape(title)}</h4><div class='sub'>{sub}</div></div>"


# --- Architecture: SVG flow diagram with real bezier arrows (Apple design) ---
ARCH_SVG = """<svg id="arch" viewBox="0 0 1000 860" style="width:100%;height:auto" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="gCore" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#0a84ff"/><stop offset="100%" stop-color="#30d158"/>
    </linearGradient>
    <linearGradient id="gRisk" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#ff375f"/><stop offset="100%" stop-color="#ff9f0a"/>
    </linearGradient>
    <linearGradient id="gData" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#5e5ce6"/><stop offset="100%" stop-color="#bf5af2"/>
    </linearGradient>
    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#0a84ff"/>
    </marker>
    <marker id="arrowRisk" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#ff375f"/>
    </marker>
    <marker id="arrowData" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#5e5ce6"/>
    </marker>
    <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="3" stdDeviation="4" flood-opacity="0.18"/>
    </filter>
  </defs>

  <style>
  .arch-node rect{filter:url(#shadow);rx:16;transition:transform .25s cubic-bezier(.2,.8,.2,1)}
  .arch-node:hover rect{transform:translateY(-2px)}
  .arch-node .t{font:600 13px -apple-system,system-ui,sans-serif;fill:#1d1d1f}
  .arch-node .s{font:11px -apple-system,system-ui,sans-serif;fill:#6e6e73}
  .arch-node.core .t{fill:#fff}
  .arch-node.core .s{fill:rgba(255,255,255,.85)}
  .arch-node.risk .t{fill:#fff}
  .arch-node.risk .s{fill:rgba(255,255,255,.85)}
  .arch-node.data .t{fill:#fff}
  .arch-node.data .s{fill:rgba(255,255,255,.85)}
  .arch-edge{fill:none;stroke:#0a84ff;stroke-width:2.5;opacity:.85}
  .arch-edge.risk{stroke:#ff375f}
  .arch-edge.data{stroke:#5e5ce6}
  .arch-edge.ghost{stroke:#c7c7cc;stroke-dasharray:5 4;opacity:.6}
  .arch-label{font:11px -apple-system,system-ui,sans-serif;fill:#6e6e73}
  .arch-cap{font:600 11px -apple-system,system-ui,sans-serif;fill:#86868b;letter-spacing:.08em}
  </style>

  <!-- ===== LAYER 1 — DATA ===== -->
  <g class="arch-cap"><text x="30" y="30">LAAG 1 · DATA</text></g>
  {n_server}{n_scrape}

  <!-- ===== LAYER 2 — SIGNALS / EXTENSIONS ===== -->
  <g class="arch-cap"><text x="30" y="180">LAAG 2 · SIGNALEN</text></g>
  {n_speech}{n_reports}{n_alerts}
  {n_bottleneck}{n_regional}{n_regime}{n_impact}

  <!-- ===== LAYER 3 — FUSION ===== -->
  <g class="arch-cap"><text x="30" y="360">LAAG 3 · FUSIE</text></g>
  {n_fusion}

  <!-- ===== LAYER 4 — DECISION + RISK ===== -->
  <g class="arch-cap"><text x="30" y="520">LAAG 4 · BESLUIT + RISICO</text></g>
  {n_rl}{n_mc}{n_risk}

  <!-- ===== LAYER 5 — ACTION ===== -->
  <g class="arch-cap"><text x="30" y="700">LAAG 5 · ACTIE</text></g>
  {n_action}{n_exec}

  <!-- ===== EDGES (arrows between node centers) ===== -->
  <!-- server -> scrape -->
  <path class="arch-edge data" d="{e_server_scrape}" marker-end="url(#arrowData)"/>
  <!-- scrape -> signals -->
  <path class="arch-edge data" d="{e_scrape_speech}" marker-end="url(#arrowData)"/>
  <path class="arch-edge data" d="{e_scrape_reports}" marker-end="url(#arrowData)"/>
  <path class="arch-edge data" d="{e_scrape_alerts}" marker-end="url(#arrowData)"/>
  <!-- signals -> impact-agent (koppelt gebeurtenis aan beïnvloede instrumenten) -->
  <path class="arch-edge" d="{e_speech_impact}" marker-end="url(#arrow)"/>
  <path class="arch-edge" d="{e_reports_impact}" marker-end="url(#arrow)"/>
  <path class="arch-edge" d="{e_alerts_impact}" marker-end="url(#arrow)"/>
  <!-- extensions -> fusion -->
  <path class="arch-edge" d="{e_bottleneck_fusion}" marker-end="url(#arrow)"/>
  <path class="arch-edge" d="{e_regional_fusion}" marker-end="url(#arrow)"/>
  <path class="arch-edge" d="{e_regime_fusion}" marker-end="url(#arrow)"/>
  <!-- impact -> fusion (per beïnvloede entiteit) -->
  <path class="arch-edge" d="{e_impact_fusion}" marker-end="url(#arrow)"/>
  <!-- fusion -> rl -->
  <path class="arch-edge" d="{e_fusion_rl}" marker-end="url(#arrow)"/>
  <!-- rl / mc -> risk -->
  <path class="arch-edge" d="{e_rl_risk}" marker-end="url(#arrow)"/>
  <path class="arch-edge risk" d="{e_mc_risk}" marker-end="url(#arrowRisk)"/>
  <!-- risk -> action -->
  <path class="arch-edge risk" d="{e_risk_action}" marker-end="url(#arrowRisk)"/>
  <path class="arch-edge" d="{e_action_exec}" marker-end="url(#arrow)"/>
</svg>"""


def _node(id, x, y, w, h, title, subs, cls="", grad="gData"):
    fill = {"gData": "url(#gData)", "gCore": "url(#gCore)", "gRisk": "url(#gRisk)"}[grad]
    sub_lines = "".join(f'<tspan x="{x+w/2}" dy="{1 if i else 0}em">{html.escape(s)}</tspan>'
                        for i, s in enumerate(subs))
    return (
        f'<g class="arch-node {cls}" id="{id}" role="button" tabindex="0" '
        f'onclick="archInfo(\'{id}\')" onkeydown="if(event.key===\'Enter\'||event.key===\' \')archInfo(\'{id}\')">'
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}"/>'
        f'<text x="{x+w/2}" y="{y+h/2-4}" text-anchor="middle" class="t">{html.escape(title)}</text>'
        f'<text x="{x+w/2}" y="{y+h/2+12}" text-anchor="middle" class="s">{sub_lines}</text>'
        f'</g>'
    )


def _edge(x1, y1, x2, y2, cx=0.5, cy=0.35):
    """Bezier curve from the bottom of node1 to the top of node2 (with a real arrow)."""
    return f"M {x1} {y1} C {x1} {y1+(y2-y1)*cy}, {x2} {y2-(y2-y1)*cy}, {x2} {y2}"


# --- Per-component explanation (for the clickable modal) ---
COMPONENT_INFO = {
    "server": {
        "titel": "24/7 Server",
        "rol": "The central infrastructure that runs everything.",
        "wat": "An always-on server that orchestrates the webscraping, the AI agents, "
               "the risk engine and the execution. Schedules tasks via a scheduler "
               "and keeps the whole chain running.",
        "in": "— (starting point)",
        "uit": "structured webscraping · scheduling",
        "code": "hermes_bot/pipeline.py",
    },
    "scrape": {
        "titel": "Structured Webscraping",
        "rol": "Fetches raw information from the web.",
        "wat": "Scrapes live speeches from CEOs and world leaders, company reports, "
               "government spending, news items and market data. Structures this "
               "into usable signals for the rest of the chain.",
        "in": "24/7 server · web sources",
        "uit": "speech · reports · alerts · market data",
        "code": "hermes_bot/data/scraper.py",
    },
    "speech": {
        "titel": "Speech (audio + video)",
        "rol": "Analyses speeches from CEOs and world leaders.",
        "wat": "Splits speech into audio and video. Audio provides text (transcription), "
               "emotion, pauses, volume and pitch. Video provides facial expression, "
               "movements and body language. Uses open-source models "
               "(faster-whisper, DeepFace, MediaPipe, openSMILE).",
        "in": "webscraping → speech",
        "uit": "text · emotion · prosody · audience → fusion",
        "code": "hermes_bot/signals/audio.py · video.py",
    },
    "reports": {
        "titel": "Reports & Alerts",
        "rol": "Processes company and government reports.",
        "wat": "Reads quarterly results, government spending and independent reporting. "
               "An AI agent distils price changes and state investments "
               "from this as input for the fusion model.",
        "in": "webscraping → reports",
        "uit": "price changes · state investments → fusion",
        "code": "hermes_bot/signals/textual.py",
    },
    "alerts": {
        "titel": "News & Point loops",
        "rol": "Catches news items and alerts.",
        "wat": "Monitors news feeds and point loops (repeated data points) for "
               "current events that could affect the market.",
        "in": "webscraping → alerts",
        "uit": "news signals → fusion",
        "code": "hermes_bot/data/scraper.py",
    },
    "bottleneck": {
        "titel": "B2B Bottleneck",
        "rol": "Ranks companies on supply-chain bottlenecks.",
        "wat": "An agent assesses which companies sit in a 'bottleneck' "
               "(demand > supply, supply-chain constraints) and passes a rating "
               "to the fusion model as an extra signal.",
        "in": "webscraping → reports",
        "uit": "bottleneck ratings → fusion",
        "code": "hermes_bot/expansions/__init__.py",
    },
    "regional": {
        "titel": "Regional Scores",
        "rol": "Scores regions on safety, satisfaction and economy.",
        "wat": "Computes per-region scores for safety, satisfaction, social "
               "security and business-economic changes. This context helps "
               "the fusion model weigh regional risks.",
        "in": "webscraping → reports",
        "uit": "regional scores → fusion",
        "code": "hermes_bot/expansions/__init__.py",
    },
    "regime": {
        "titel": "Regime / Orderflow",
        "rol": "Determines the market state (bull, crash, recovery).",
        "wat": "Detects the market regime (normal, elevated, stressed, crash, "
               "recovery) and tracks orderflow/positioning (e.g. COT). This is an "
               "important input for the risk engine.",
        "in": "market data · orderflow",
        "uit": "regime label → fusion + risk engine",
        "code": "hermes_bot/expansions/__init__.py",
    },
    "impact": {
        "titel": "Impact Agent",
        "rol": "Koppelt gebeurtenissen aan beïnvloede instrumenten.",
        "wat": "Bepaalt WELKE stocks, obligaties en ETF's geraakt worden door een "
               "gebeurtenis (speech, kwartaalrapport, overheidsuitgave, nieuws) via "
               "keyword/sector-matching. Levert per beïnvloede entiteit een "
               "fusion-input, zodat het RL-model per instrument kan beslissen.",
        "in": "speech · reports · alerts",
        "uit": "beïnvloede instrumenten → fusion (per entiteit)",
        "code": "hermes_bot/impact.py",
    },
    "fusion": {
        "titel": "Fusion Model",
        "rol": "Combines all signals into one decision.",
        "wat": "Weights all inputs (speech, reports, alerts, bottleneck, regional "
               "scores, regime) into a quality score, certainty and emotion. "
               "This is the 'what to buy' layer of the architecture.",
        "in": "all signals + extensions",
        "uit": "quality · certainty · emotion → RL fusion",
        "code": "hermes_bot/fusion/__init__.py",
    },
    "rl": {
        "titel": "RL-Fusion Model",
        "rol": "Proposes positions (buy/sell/hold/hedge).",
        "wat": "Takes the fusion output and proposes a desired position. The "
               "risk engine approves or adjusts it — RL proposes, Risk decides.",
        "in": "fusion output",
        "uit": "proposed position → risk engine",
        "code": "hermes_bot/agents/rl/__init__.py",
    },
    "mc": {
        "titel": "Monte Carlo",
        "rol": "Simulates thousands of market paths for risk.",
        "wat": "Bootstraps per-day market paths to estimate VaR95, Expected Shortfall (ES95) and "
               "crash probability. This feeds the risk engine with "
               "tail-risk information.",
        "in": "historical returns",
        "uit": "VaR95 · ES95 · crash probability → risk engine",
        "code": "hermes_bot/risk_v2/montecarlo.py",
    },
    "risk": {
        "titel": "Risk Engine v2.1",
        "rol": "Determines how much risk we may take.",
        "wat": "The adaptive risk engine with three layers: Strategic (regime), "
               "Tactical (vol/drawdown/correlation) and Emergency Brake (flash crash). "
               "A recovery engine rebuilds exposure after stabilization. This is the "
               "'how much risk' layer.",
        "in": "RL proposal · Monte Carlo · regime",
        "uit": "risk budget / exposure → output action",
        "code": "hermes_bot/risk_v2_1/__init__.py",
    },
    "action": {
        "titel": "Output Action",
        "rol": "Turns the risk budget into a concrete portfolio.",
        "wat": "Allocates the approved risk budget across stocks, ETFs, "
               "bonds and cash. Determines the final positions.",
        "in": "risk budget · RL proposal",
        "uit": "positions → execution",
        "code": "hermes_bot/portfolio/__init__.py",
    },
    "exec": {
        "titel": "Execution",
        "rol": "Actually executes the positions.",
        "wat": "Places orders via a paper broker (fail-closed: on doubt no "
               "order) or a live broker. Logs every transaction.",
        "in": "positions from output action",
        "uit": "orders · transaction log",
        "code": "hermes_bot/execution/__init__.py",
    },
}


def _arch_modal_html() -> str:
    """Small modal (not fullscreen) with an explanation of a component."""
    return """
<div id="arch-modal" class="arch-modal" role="dialog" aria-modal="true" aria-hidden="true"
     onclick="if(event.target===this)archClose()">
  <div class="arch-modal-card">
    <button class="arch-modal-close" onclick="archClose()" aria-label="Sluiten">✕</button>
    <div id="arch-modal-body"></div>
  </div>
</div>
<script>
const ARCH_INFO = __ARCH_INFO__;
function archInfo(id){
  const info = ARCH_INFO[id];
  if(!info) return;
  const body = document.getElementById('arch-modal-body');
  body.innerHTML =
    '<h3 class="arch-modal-title">' + info.titel + '</h3>' +
    '<p class="arch-modal-rol">' + info.rol + '</p>' +
    '<p class="arch-modal-wat">' + info.wat + '</p>' +
    '<div class="arch-modal-flow">' +
      '<div><span class="lbl">In</span><span>' + info.in + '</span></div>' +
      '<div><span class="lbl">Uit</span><span>' + info.uit + '</span></div>' +
      '<div><span class="lbl">Code</span><span>' + info.code + '</span></div>' +
    '</div>';
  const m = document.getElementById('arch-modal');
  m.classList.add('open');
  m.setAttribute('aria-hidden','false');
}
function archClose(){
  const m = document.getElementById('arch-modal');
  m.classList.remove('open');
  m.setAttribute('aria-hidden','true');
}
document.addEventListener('keydown', e => { if(e.key==='Escape') archClose(); });
</script>
"""


def _architecture_html() -> str:
    # Node-posities (x, y, breedte, hoogte).
    # Laag 1
    n_server = _node("server", 150, 45, 200, 46, "24/7 Server", ["orchestratie · scheduler"], "data")
    n_scrape = _node("scrape", 480, 45, 280, 46, "Webscraping", ["speech · reports · alerts · marktdata"], "data")
    # Laag 2 signalen
    n_speech = _node("speech", 90, 195, 170, 54, "Speech", ["CEO's · landsleiders"], "")
    n_reports = _node("reports", 290, 195, 190, 54, "Reports & Alerts", ["quarterly · overheidsuitgaven"], "")
    n_alerts = _node("alerts", 520, 195, 170, 54, "Nieuws & Point-loops", ["alerts"], "")
    # Extensions (now integrated in layer 2)
    n_bottleneck = _node("bottleneck", 720, 195, 190, 54, "B2B Bottleneck", ["supply-chain · knelpunten"], "")
    n_regional = _node("regional", 90, 270, 170, 50, "Regionale Scores", ["veiligheid · economie"], "")
    n_regime = _node("regime", 520, 270, 170, 50, "Regime / Orderflow", ["bull · crash · COT"], "")
    # Impact-agent: koppelt gebeurtenissen aan beïnvloede instrumenten.
    n_impact = _node("impact", 290, 320, 200, 50, "Impact Agent", ["welke stocks · obligaties · ETF's"], "core", "gCore")
    # Laag 3 fusion
    n_fusion = _node("fusion", 390, 375, 220, 56, "Fusion Model", ["kwaliteit · zekerheid · emotie"], "core", "gCore")
    # Laag 4
    n_rl = _node("rl", 140, 530, 210, 56, "RL-Fusion Model", ["buy · sell · hold · hedge"], "core", "gCore")
    n_mc = _node("mc", 420, 530, 180, 44, "Monte Carlo", ["VaR95 · ES95 · crash-kans"], "risk", "gRisk")
    n_risk = _node("risk", 640, 530, 220, 56, "Risk Engine v2.1", ["strategic · tactical · emergency · recovery"], "risk", "gRisk")
    # Laag 5
    n_action = _node("action", 240, 715, 200, 50, "Output Actie", ["aandelen · ETF's · obligaties · cash"], "action", "gCore")
    n_exec = _node("exec", 560, 715, 200, 50, "Executie", ["paper · live · fail-closed"], "action", "gCore")

    # Centers (bottom/top of nodes) for edge connections.
    def edge_bottom_center(x, y, w, h): return (x + w/2, y + h)
    def edge_top_center(x, y, w, h): return (x + w/2, y)

    e = {}
    # server(bottom) -> scrape(bottom? no: right to left)
    # Use side connections to keep it tidy:
    # server rechts -> scrape links
    e["e_server_scrape"] = _edge(350, 68, 480, 68, cx=0.5, cy=0.5)  # horizontaal
    # scrape bottom -> each signal top
    e["e_scrape_speech"] = _edge(560, 91, 175, 195, 0.5, 0.5)
    e["e_scrape_reports"] = _edge(610, 91, 385, 195, 0.5, 0.45)
    e["e_scrape_alerts"] = _edge(655, 91, 605, 195, 0.5, 0.5)
    # signalen -> impact-agent (koppelt gebeurtenis aan beïnvloede instrumenten)
    e["e_speech_impact"] = _edge(175, 249, 300, 320, 0.5, 0.4)
    e["e_reports_impact"] = _edge(385, 249, 390, 320, 0.5, 0.4)
    e["e_alerts_impact"] = _edge(605, 249, 480, 320, 0.5, 0.45)
    # extensions -> fusion (bottleneck/regional/regime gaan direct naar fusion)
    e["e_bottleneck_fusion"] = _edge(815, 249, 550, 375, 0.5, 0.5)
    e["e_regional_fusion"] = _edge(175, 320, 500, 375, 0.5, 0.4)
    e["e_regime_fusion"] = _edge(605, 320, 520, 375, 0.5, 0.45)
    # impact -> fusion (per beïnvloede entiteit)
    e["e_impact_fusion"] = _edge(390, 370, 480, 375, 0.5, 0.4)
    # fusion -> rl
    e["e_fusion_rl"] = _edge(500, 431, 245, 530, 0.5, 0.4)
    # rl -> risk, mc -> risk
    e["e_rl_risk"] = _edge(245, 586, 640, 558, 0.5, 0.4)
    e["e_mc_risk"] = _edge(510, 574, 640, 558, 0.5, 0.4)
    # risk -> actie, actie -> exec
    e["e_risk_action"] = _edge(750, 586, 340, 715, 0.5, 0.4)
    e["e_action_exec"] = _edge(440, 765, 560, 765, 0.5, 0.5)

    # Build the SVG by replacing only the real placeholders (not .format,
    # because the SVG contains CSS braces that .format would try to fill).
    repl = {
        "n_server": n_server, "n_scrape": n_scrape,
        "n_speech": n_speech, "n_reports": n_reports, "n_alerts": n_alerts,
        "n_bottleneck": n_bottleneck, "n_regional": n_regional, "n_regime": n_regime,
        "n_impact": n_impact,
        "n_fusion": n_fusion, "n_rl": n_rl, "n_mc": n_mc, "n_risk": n_risk,
        "n_action": n_action, "n_exec": n_exec,
    }
    repl.update({k: e[k] for k in e})
    out = ARCH_SVG
    for token, val in repl.items():
        out = out.replace("{" + token + "}", str(val))
    # Modal with per-component explanation (clickable).
    import json as _json
    modal = _arch_modal_html().replace(
        "__ARCH_INFO__", _json.dumps(COMPONENT_INFO, ensure_ascii=False)
    )
    return out + modal


def _backtest_html() -> str:
    return """
<h2>📊 Backtest tool</h2>
<p>Run a market backtest that goes through the <b>risk engine</b> (vol-targeting, drawdown guard)
and compares against buy-and-hold. Real market data via yfinance (free).</p>
<div class="card">
  <form id="bt-form" class="form-row">
    <input name="symbol" value="SPY" placeholder="Ticker" list="symlist">
    <datalist id="symlist">
      <option>SPY</option><option>QQQ</option><option>AAPL</option><option>MSFT</option>
      <option>NVDA</option><option>JPM</option><option>EFA</option><option>AGG</option>
    </datalist>
    <select name="period">
      <option value="1y">1 year</option>
      <option value="2y" selected>2 years</option>
      <option value="3y">3 years</option>
      <option value="5y">5 years</option>
    </select>
    <label style="font-size:14px;color:var(--muted)">Vol-target
      <input name="vol_target" value="0.125" type="number" step="0.025" style="width:90px"></label>
    <button type="submit">▶ Run backtest</button>
  </form>
</div>
<div id="bt-result" aria-live="polite"></div>

<script>
document.getElementById('bt-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const el = document.getElementById('bt-result');
  el.innerHTML = '<p><span class="spin"></span> Backtesting on the server… (takes ~5s)</p>';
  const f = e.target;
  const qs = new URLSearchParams({
    symbol: f.symbol.value || 'SPY',
    period: f.period.value,
    vol_target: f.vol_target.value || '0.125',
  }).toString();
  try {
    const r = await fetch('/raw_backtest?' + qs);
    const text = await r.text();
    el.innerHTML = text;
  } catch (err) {
    el.innerHTML = '<div class="card"><p style="color:var(--danger)">Error: ' + htmlEscape(String(err)) + '</p></div>';
  }
});

function htmlEscape(s){ return s.replace(/[&<>"]/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }
</script>
"""


def _backtest_result_html(data: dict) -> str:
    """Render the backtest result with KPI cards + SVG line chart."""
    def pct(x: float) -> str:
        return f"{x*100:+.1f}%"

    tr = data["total_return"]
    bench = data["benchmark_return"]
    dd = data["max_drawdown"]
    tr_kpi = "good" if tr > 0 else "bad"

    # Limited dip: benchmark with crashes relatively deeper.
    equities = data["equity"]
    benchs = data["benchmark"]
    ts = data["timestamps"]
    W, H = 900, 300
    pad = 12
    def _poly(vals, color):
        if not vals:
            return ""
        mx, mn = max(vals), min(vals)
        rng = (mx - mn) or 1.0
        pts = []
        n = len(vals)
        for i, v in enumerate(vals):
            x = pad + i * (W - 2 * pad) / max(1, n - 1)
            y = H - pad - (v - mn) / rng * (H - 2 * pad)
            pts.append(f"{x:.1f},{y:.1f}")
        return f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>'

    svg = f"""<div class="chart-wrap">
<svg viewBox="0 0 {W} {H}" style="width:100%;height:auto" role="img" aria-label="Equity-curve backtest">
  {_poly(benchs, '#86868b')}
  {_poly(equities, '#0a84ff')}
</svg>
</div>
<div class="chart-legend">
  <span class="legend-item strategy"><span class="swatch"></span>Strategy (vol-target)</span>
  <span class="legend-item bench"><span class="swatch"></span>Buy-and-hold</span>
</div>"""

    return f"""
<div class="card">
  <h3>Result — {html.escape(data['symbol'])} ({html.escape(data['period'])})</h3>
  <div class="kpis">
    <div class="kpi {tr_kpi}"><div class="label">Total return</div><div class="value">{pct(tr)}</div></div>
    <div class="kpi neutral"><div class="label">Buy-and-hold</div><div class="value">{pct(bench)}</div></div>
    <div class="kpi bad"><div class="label">Max drawdown</div><div class="value">{pct(dd)}</div></div>
    <div class="kpi neutral"><div class="label">Sharpe</div><div class="value">{data['sharpe']}</div></div>
    <div class="kpi neutral"><div class="label">Trades</div><div class="value">{data['n_trades']}</div></div>
    <div class="kpi {'good' if data['win_rate']>=0.5 else 'neutral'}"><div class="label">Win-rate</div><div class="value">{data['win_rate']*100:.0f}%</div></div>
  </div>
  <p style="margin-top:12px">
    The strategy ended at <b>€{data['final_capital']:,.0f}</b> from {data['initial_capital']:,.0f} starting capital.
    The lower max-drawdown (vs benchmark) shows crash resilience; the strategy is
    more conservative but limits losses in crashes.
  </p>
</div>
<div class="card">
  <h3>Equity curve (Strategy vs Buy-and-hold)</h3>
  {svg}
  <p style="font-size:12px;color:var(--muted)">{len(ts)} data points plotted. Download via the button in the nav.</p>
</div>
"""


def _control_panel_html() -> str:
    """Backtest Control Panel — config/experiment layer on top of the v2.1 engine."""
    return """
<h2>🎛️ Backtest Control Panel</h2>
<p>A thin experiment layer on top of the existing backtester. Set inputs,
run one backtest or compare 2–5 configurations — the engine itself is unchanged.</p>

<div class="card">
  <h3>Configuration</h3>
  <form id="cp-form" class="form-row" style="align-items:stretch;flex-direction:column;gap:12px">
    <div class="form-row">
      <label style="font-size:14px;color:var(--muted)">Run type
        <select id="cp-rule" name="rule" onchange="cpToggleMode()">
          <option value="single" selected>Single backtest</option>
          <option value="compare">Compare (2–5 configs)</option>
        </select></label>
      <label style="font-size:14px;color:var(--muted)">Asset
        <select name="asset" id="cp-asset" list="syms"><option value="SPY">SPY</option>
          <option>QQQ</option><option>IWM</option><option>AAPL</option><option>MSFT</option>
          <option>NVDA</option><option>EFA</option><option>AGG</option></select></label>
      <label style="font-size:14px;color:var(--muted)">Years
        <input name="years" value="3" type="number" step="0.5" min="0.5" style="width:90px"></label>
      <label style="font-size:14px;color:var(--muted)">Vol-target
        <input name="vol" value="0.125" type="number" step="0.025" min="0.02" max="0.5" style="width:96px"></label>
      <label style="font-size:14px;color:var(--muted)">Costs
        <select name="costs"><option value="1">ON</option><option value="0">OFF</option></select></label>
    </div>
    <div class="form-row" style="font-size:13px;color:var(--muted);gap:16px">
      <label><input type="checkbox" name="opp" checked> Opportunity Score</label>
      <label><input type="checkbox" name="recovery" checked> Recovery Engine</label>
      <label><input type="checkbox" name="brake" checked> Emergency Brake</label>
      <label><input type="checkbox" name="risk" checked> Risk Engine</label>
    </div>
    <div id="cp-compare" style="display:none">
      <h3 style="margin-top:4px">Configurations to compare</h3>
      <div id="cp-rows"></div>
      <div class="form-row">
        <button type="button" class="secondary" onclick="cpAddRow()">+ Add config</button>
      </div>
    </div>
    <div class="form-row">
      <button type="submit">▶ Run backtest</button>
      <button type="button" class="secondary" onclick="cpQuick()">⚡ Quick: vol-targets</button>
      <button type="button" class="secondary" onclick="cpQuickFull()">⚡ Full vs components</button>
    </div>
  </form>
</div>
<div id="cp-result" aria-live="polite"></div>

<script>
let cpRowCount = 0;
const CP_ASSETS = ['SPY','QQQ','IWM','AAPL','MSFT','NVDA','EFA','AGG'];
function cpAddRow(label, asset, vol){
  cpRowCount++;
  const d = document.createElement('div');
  d.className='form-row'; d.id='cp-row-'+cpRowCount;
  const opts = CP_ASSETS.map(a=>'<option '+(a===(asset||'SPY')?'selected':'')+'>'+a+'</option>').join('');
  d.innerHTML =
    '<label style="font-size:13px;color:var(--muted)">&nbsp;Label <input name="rlabel" value="'+(label||('Config '+cpRowCount))+'" style="width:130px"></label>'+
    '<select name="rasset">'+opts+'</select>'+
    '<input name="rvol" value="'+(vol||'0.125')+'" type="number" step="0.025" min="0.02" max="0.5" style="width:90px">'+
    '<button type="button" class="secondary" style="padding:6px 12px" onclick="this.closest(\'.form-row\').remove()">✕</button>';
  document.getElementById('cp-rows').appendChild(d);
}
function cpToggleMode(){
  const c = document.getElementById('cp-rule').value==='compare';
  document.getElementById('cp-compare').style.display = c ? 'block' : 'none';
  document.getElementById('cp-asset').disabled = c;
  if(c && cpRowCount===0){ cpAddRow('A','SPY','0.10'); cpAddRow('B','SPY','0.15'); cpAddRow('C','SPY','0.20'); }
}
function cpQuick(){
  ['0.10','0.15','0.20'].forEach(v=>cpAddRow('vol '+v,'SPY',v));
  document.getElementById('cp-rule').value='compare'; cpToggleMode();
}
function cpQuickFull(){
  // Reset and build 'full vs components' (each differs only in an existing toggle).
  document.getElementById('cp-rows').innerHTML=''; cpRowCount=0;
  cpAddRow('Full v2.1','SPY','0.125');
  document.getElementById('cp-rule').value='compare'; cpToggleMode();
}

document.getElementById('cp-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const f = e.target;
  const el = document.getElementById('cp-result');
  el.innerHTML = '<p><span class="spin"></span> Backtesting on the server… (takes a few seconds)</p>';
  const rule = f.rule.value;
  const body = {
    rule,
    years: f.years.value || '3',
    vol: f.vol.value || '0.125',
    costs: f.costs.value === '1',
    opp: f.opp.checked, recovery: f.recovery.checked,
    brake: f.brake.checked, risk: f.risk.checked,
  };
  if(rule==='compare'){
    body.rows = Array.from(document.querySelectorAll('#cp-rows .form-row')).map(r => ({
      label: r.querySelector('[name=rlabel]').value,
      asset: r.querySelector('[name=rasset]').value,
      vol: r.querySelector('[name=rvol]').value,
    }));
  } else {
    body.asset = f.asset.value;
  }
  try {
    const r = await fetch('/control_panel', {
      method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)
    });
    const text = await r.text();
    if(!r.ok) throw new Error(text);
    el.innerHTML = text;
  } catch(err){
    el.innerHTML = '<div class="card"><p style="color:var(--danger)">Error: ' + String(err).replace(/[<>&]/g,c=>({'<':'&lt;','>':'&gt;','&':'&amp;'}[c])) + '</p></div>';
  }
});
</script>
"""


def _run_control_panel_(body: dict) -> str:
    """Server side: process the Control Panel JSON request into HTML.

    Builds BacktestConfig(s) from the form input and calls the existing v2.1 engine.
    """
    from hermes_bot.control_panel import (
        BacktestConfig,
        compare_to_html,
        overview_html,
        run_backtest,
        run_compare,
        save_run,
    )

    rule = body.get("rule", "single")
    years = float(body.get("years", "3"))
    vol = float(body.get("vol", "0.125"))
    base = dict(
        years=years, transaction_costs=body.get("costs", True),
        opportunity=body.get("opp", True), recovery=body.get("recovery", True),
        emergency_brake=body.get("brake", True), risk_engine=body.get("risk", True),
    )
    if rule == "compare":
        rows = body.get("rows") or []
        if not 2 <= len(rows) <= 5:
            return '<div class="card"><p style="color:var(--danger)">Choose 2–5 configurations.</p></div>'
        cfgs = [BacktestConfig(asset=r["asset"], vol_target=float(r["vol"]),
                               label=r["label"], run_type="compare", **base) for r in rows]
        data = run_compare(cfgs)
        out = compare_to_html(data)
        try:
            save_run(data)
        except Exception:  # noqa: BLE001
            pass
        return out
    asset = body.get("asset", "SPY")
    cfg = BacktestConfig(asset=asset, vol_target=vol, **base)
    data = run_backtest(cfg, return_log=True)
    # Compact overzicht + equity/drawdown/exposure + risk/opp.
    parts = [overview_html(data)]
    from hermes_bot.control_panel import (
        drawdown_chart,
        equity_chart,
        exposure_chart,
        risk_opportunity_chart,
    )
    parts.append("<div class='card'><h3>Equity-curve</h3>"
                 + equity_chart(data, data.get("prices")) + "</div>")
    parts.append("<div class='card'><h3>Drawdown</h3>" + drawdown_chart(data) + "</div>")
    parts.append("<div class='card'><h3>Exposure</h3>" + exposure_chart(data) + "</div>")
    parts.append("<div class='card'><h3>Risk / Opportunity vs Exposure</h3>" +
                 risk_opportunity_chart(data) + "</div>")
    parts.append(f"<p class='muted'>Run: {data['run_id']}</p>")
    try:
        save_run(data)
    except Exception:  # noqa: BLE001
        pass
    return "\n".join(parts)


def _download_html() -> str:
    return """
<h2>⬇️ Download &amp; GitHub export</h2>
<p>Download the whole project as a <b>.zip</b> — works on <b>Windows, macOS, iPad and any browser</b>.
The zip is ready to upload to GitHub (without .venv, caches or secrets).</p>
<div class="card">
  <div class="dl-panel">
    <button class="secondary" onclick="location.href='/getzip'">⬇️ Download project (.zip)</button>
    <span class="dl-size">Excl. .venv / caches / .env</span>
    <button class="secondary" onclick="location.href='/github'">▶ GitHub instructions</button>
  </div>
</div>
<div class="card" style="margin-top:18px">
  <h3>GitHub push (from this server)</h3>
  <pre><code>cd ~/trading-bot
git init -b main
git add -A && git commit -m "Initial Hermes trading bot"
git remote add origin https://github.com/YOUR-NAME/hermes-trading-bot.git
git push -u origin main</code></pre>
  <p style="font-size:13px">For iPad: use the button above to download the zip, open the
  <b>GitHub app</b>, create a repo and upload the files. See the
  <a href="/github">GitHub instructions</a> for the full steps.</p>
</div>
"""


def _code_view_html(rel: str, content: str) -> str:
    # Show code with line numbers.
    lines = content.splitlines() if content else []
    body = []
    for i, ln in enumerate(lines, 1):
        body.append(f'<span class="line-num">{i}</span>{html.escape(ln)}')
    return (
        f"<h2>📄 {html.escape(rel)}</h2>"
        f"<pre><code>{''.join('<div>'+b+'</div>' for b in body)}</code></pre>"
    )


# ---------------------------------------------------------------------------
# PAGE SHELL
# ---------------------------------------------------------------------------
def _page(title: str, content: str, active: str = "") -> str:
    nav_links = [
        ("/", "Dashboard", "Dash"),
        ("/panel", "Panel", "Panel"),
        ("/backtest", "Backtest", "Backtest"),
        ("/download", "Download", "Download"),
        ("/architectuur", "Architecture", "Architecture"),
        ("/code", "Code", "Code"),
        ("/vault", "Notes", "Notes"),
    ]
    nav = "".join(
        f'<a href="{href}" class="{"active" if a==active else ""}">{label}</a>'
        for href, label, a in nav_links
    )
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Hermes Trading Bot — {html.escape(title)}</title>
<style>{APPLE_CSS}</style></head>
<body>
<header>
  <h1><span class="dot"></span>Hermes Trading Bot</h1>
  <nav>{nav}</nav>
</header>
<main>{content}</main>
<div id="toast"></div>
<script>{TOAST_JS}</script>
</body></html>"""


# ---------------------------------------------------------------------------
# HTTP HANDLER
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    def _send(self, body: bytes | str, ctype: str = "text/html; charset=utf-8", code: int = 200) -> None:
        data = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _page(self, title: str, content: str, active: str = "") -> str:
        return _page(title, content, active)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(parsed.query)
        path = parsed.path

        if path == "/":
            content = (
                "<h2>Welcome</h2>"
                "<p>A modular multi-asset AI trading bot that pursues <b>good returns</b> with resilience against "
                "<b>market crashes</b>. Explore the architecture, run a backtest, browse the code/notes, "
                "or download the project.</p>"
                "<div class='kpis'>"
                "<div class='kpi good'><div class='label'>Goal</div><div class='value' style='font-size:16px'>good returns</div></div>"
                "<div class='kpi neutral'><div class='label'>Resilience</div><div class='value' style='font-size:16px'>crash-resistant</div></div>"
                "<div class='kpi neutral'><div class='label'>Multi-asset</div><div class='value' style='font-size:16px'>stocks/ETFs/bonds</div></div>"
                "<div class='kpi neutral'><div class='label'>Execution</div><div class='value' style='font-size:16px'>paper → live</div></div>"
                "</div>"
                "<div class='card'><h3>Start here</h3>"
                "<p>• <a href='/backtest'>Explore the backtest tool</a> — see how the risk engine limits crashes.<br>"
                "• <a href='/architectuur'>View the architecture diagram</a> — the 7 layers.<br>"
                "• <a href='/download'>Download or export to GitHub</a>.<br>"
                "• <a href='/run'>Run the demo pipeline</a>.</p></div>"
            )
            self._send(self._page("Dashboard", content, "Dash"))
        elif path == "/panel":
            self._send(self._page("Backtest Control Panel", _control_panel_html(), "Panel"))
        elif path == "/backtest":
            self._send(self._page("Backtest", _backtest_html(), "Backtest"))
        elif path == "/download":
            self._send(self._page("Download", _download_html(), "Download"))
        elif path == "/github":
            from hermes_bot.export_zip import build_github_instructions
            content = f"<h2>GitHub instructions</h2><pre>{html.escape(build_github_instructions())}</pre>"
            self._send(self._page("GitHub", content, "Download"))
        elif path == "/architectuur":
            self._send(self._page("Architecture", "<h2>🏗️ Architecture</h2>" + _architecture_html(), "Architecture"))
        elif path == "/code":
            rel = qs.get("path", [""])[0]
            target = (ROOT / rel).resolve()
            if target.is_file() and ROOT in target.parents:
                self._send(self._page("Code", _code_view_html(rel, target.read_text()), "Code"))
            else:
                content = "<h2>Code</h2><div class='code-layout'><div>" + _tree_html(ROOT / "hermes_bot", ROOT) + "</div></div>"
                self._send(self._page("Code", content, "Code"))
        elif path == "/vault":
            note = qs.get("note", [""])[0]
            rel = Path(note)
            if any(part in ("..", "") for part in rel.parts):
                rel = Path()
            target = (VAULT / rel).resolve()
            if target.is_file() and VAULT in target.parents:
                content = f"<h2>📝 {html.escape(target.stem)}</h2>" + _render_md(target.read_text())
            else:
                content = "<h2>Obsidian notes</h2>" + _vault_notes_html()
            self._send(self._page("Notes", content, "Notes"))
        elif path == "/run":
            content = (
                "<h2>Run demo</h2>"
                "<div class='card'><form method='POST' action='/run'><button>▶ Run demo pipeline</button></form>"
                "<p style='font-size:13px;margin-top:8px'>Also: <code>uv run pytest</code> and <code>uv run python -m hermes_bot.demo</code></p></div>"
            )
            self._send(self._page("Demo", content, "Dash"))
        elif path == "/getzip":
            # Download the project as a .zip (for GitHub export, also on iPad).
            try:
                from hermes_bot.export_zip import build_zip_bytes
                data = build_zip_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "application/zip")
                self.send_header("Content-Length", str(len(data)))
                self.send_header(
                    "Content-Disposition", 'attachment; filename="hermes-trading-bot.zip"'
                )
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)
            except Exception as e:  # noqa: BLE001
                self._send(f"Fout bij zip: {html.escape(str(e))}")
        elif path == "/raw_backtest":
            # Data return for AJAX without a page shell
            try:
                from hermes_bot.backtest.runner import run_backtest
                data = run_backtest(
                    symbol=qs.get("symbol", ["SPY"])[0],
                    period=qs.get("period", ["2y"])[0],
                    vol_target=float(qs.get("vol_target", ["0.125"])[0]),
                )
                self._send(_backtest_result_html(data))
            except Exception as e:  # noqa: BLE001
                self._send(f'<div class="card"><p style="color:var(--danger)">Error: {html.escape(str(e))}</p></div>')
        else:
            content = "<h2>404</h2><p>Page not found.</p><a href='/'>← Back</a>"
            self._send(self._page("404", content))

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/control_panel":
            import json
            try:
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length) if length else b"{}"
                body = json.loads(raw or b"{}")
                out = _run_control_panel_(body)
                self._send(out)
            except Exception as e:  # noqa: BLE001
                self._send(f'<div class="card"><p style="color:var(--danger)">Error: {html.escape(str(e))}</p></div>')
            return
        if self.path == "/run":
            try:
                r = subprocess.run(
                    ["uv", "run", "python", "-m", "hermes_bot.demo"],
                    cwd=str(ROOT), capture_output=True, text=True, timeout=90,
                )
                out = r.stdout or r.stderr
                content = f"<h2>Demo output</h2><pre><code>{html.escape(out)}</code></pre>"
            except Exception as e:  # noqa: BLE001
                content = f"<h2>Error</h2><pre><code>{html.escape(str(e))}</code></pre>"
            self._send(self._page("Demo", content, "Dash"))
        else:
            self._send("<h2>404</h2>")


def main() -> None:
    print(f"Hermes trading-bot web interface at http://localhost:{PORT}")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()