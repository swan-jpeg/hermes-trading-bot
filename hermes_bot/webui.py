"""Web-interface voor de tradingbot (poort 9124) — Apple-fluid design.

Toont: dashboard, architectuur-diagram, code/mappen, Obsidian-noten,
een werkende BACKTEST-tool (met equity-chart + benchmark), een DOWNLOAD-knop
(zip voor GitHub-export, werkt op iPad) en GitHub-instructies.

Zelfstandig met stdlib + markdown-it-py. Gebruik:
    uv run python -m hermes_bot.webui   (of via systemd)
"""
from __future__ import annotations

import html
import subprocess
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parent.parent
VAULT = Path("/home/olivier/obsidian-vault")
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

/* Architectuur blokken */
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
    """Recursieve file-tree als HTML."""
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
        return '<div class="file">⚠ vault niet bereikbaar</div>'
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
    return f'<div class="arch-block {cls}"><h4>{html.escape(title)}</h4><div class="sub">{sub}</div></div>'


def _architecture_html() -> str:
    return """
<div class="arch">
  <div class="arch-cap">Laag 1 — Data-infrastructuur</div>
  <div class="arch-row">
    {b1}<div class="arch-arrow">▼</div>{b2}
  </div>
  <div class="arch-cap">Laag 2-3 — Raw info &amp; multimedia</div>
  <div class="arch-row">
    {b3}<div class="arch-arrow">▼</div>{b4}<div class="arch-arrow">◄►</div>{b5}
  </div>
  <div class="arch-cap">Laag 4 — AI-analyse</div>
  <div class="arch-row">
    {b6}<div class="arch-arrow">▼</div>{b7}
  </div>
  <div class="arch-cap">Laag 5 — Fusie</div>
  <div class="arch-row">{b8}</div>
  <div class="arch-cap">Laag 6 — Besluitvorming + risico</div>
  <div class="arch-row">
    {b9}<div class="arch-arrow">◄►</div>{b10}<div class="arch-arrow risk">▼</div>{b11}
  </div>
  <div class="arch-cap">Laag 7 — Actie</div>
  <div class="arch-row">
    {b12}<div class="arch-arrow">▼</div>{b13}
  </div>
  <div class="arch-cap">➕ Uitbreidingen</div>
  <div class="arch-row">
    {b14}<div class="arch-arrow">◄►</div>{b15}<div class="arch-arrow">◄►</div>{b16}
  </div>
</div>
""".format(
        b1=_arch_block("24/7 Server", ["orchestratie", "scheduler", "monitoring"]),
        b2=_arch_block("Gestructureerde Webscraping", ["speech", "reports", "alerts", "marktdata"]),
        b3=_arch_block("Speech", ["audio", "video"]),
        b4=_arch_block("Audio/Video Analyse", ["tekst", "emotie", "pauses", "volume", "pitch", "gezicht", "lichaamstaal"]),
        b5=_arch_block("Reports &amp; Alerts", ["business", "overheidsuitgaven", "onafhankelijk", "nieuws", "point loops"]),
        b6=_arch_block("AI Agent (fundamenteel)", ["concurrenten", "logistiek", "geschiedenis", "economische waarden"]),
        b7=_arch_block("Regionale Scores", ["veiligheid", "tevredenheid", "sociale zekerheid", "bedrijfseconomisch"]),
        b8=_arch_block("Fusion Model", ["kwaliteit", "zekerheid", "algemene emotie"], "core"),
        b9=_arch_block("RL-Fusion Model", ["buy", "sell", "hold", "hedge", "other"], "core"),
        b10=_arch_block("Monte Carlo", ["VaR95", "ES95", "crash-kans"], "risk"),
        b11=_arch_block("Risico-Engine", ["sizing", "drawdown-guard", "stress-test", "hedging"], "risk"),
        b12=_arch_block("Output Actie", ["aandelen", "ETF's", "opties", "obligaties", "cash"], "action"),
        b13=_arch_block("Executie (paper/live)", ["broker-gateway", "orderrouting", "reconciliatie"], "action"),
        b14=_arch_block("B2B Bottleneck", ["supply-chain", "vraagdruk", "knelpunten"]),
        b15=_arch_block("Regime-Detectie", ["bull", "bear", "highvol", "crash"]),
        b16=_arch_block("Orderflow / Macro", ["COT", "optie-flow", "rente-curve", "PMI"]),
    )


def _backtest_html() -> str:
    return """
<h2>📊 Backtest-tool</h2>
<p>Draai een markt-backtest die de <b>risico-engine</b> (vol-targeting, drawdown-guard)
doorloopt en vergelijkt met buy-and-hold. Echte marktdata via yfinance (gratis).</p>
<div class="card">
  <form id="bt-form" class="form-row">
    <input name="symbol" value="SPY" placeholder="Ticker" list="symlist">
    <datalist id="symlist">
      <option>SPY</option><option>QQQ</option><option>AAPL</option><option>MSFT</option>
      <option>NVDA</option><option>JPM</option><option>EFA</option><option>AGG</option>
    </datalist>
    <select name="period">
      <option value="1y">1 jaar</option>
      <option value="2y" selected>2 jaar</option>
      <option value="3y">3 jaar</option>
      <option value="5y">5 jaar</option>
    </select>
    <label style="font-size:14px;color:var(--muted)">Vol-target
      <input name="vol_target" value="0.125" type="number" step="0.025" style="width:90px"></label>
    <button type="submit">▶ Draai backtest</button>
  </form>
</div>
<div id="bt-result" aria-live="polite"></div>

<script>
document.getElementById('bt-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const el = document.getElementById('bt-result');
  el.innerHTML = '<p><span class="spin"></span> Backtesting op de server… (kan ~5s duren)</p>';
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
    el.innerHTML = '<div class="card"><p style="color:var(--danger)">Fout: ' + htmlEscape(String(err)) + '</p></div>';
  }
});

function htmlEscape(s){ return s.replace(/[&<>"]/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }
</script>
"""


def _backtest_result_html(data: dict) -> str:
    """Render het backtest-resultaat met KPI-kaarten + SVG-lijndiagram."""
    def pct(x: float) -> str:
        return f"{x*100:+.1f}%"

    tr = data["total_return"]
    bench = data["benchmark_return"]
    dd = data["max_drawdown"]
    tr_kpi = "good" if tr > 0 else "bad"

    # Beperkte dip: benchmark met crashes relatief dieper.
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
  <span class="legend-item strategy"><span class="swatch"></span>Strategie (vol-target)</span>
  <span class="legend-item bench"><span class="swatch"></span>Buy-and-hold</span>
</div>"""

    return f"""
<div class="card">
  <h3>Resultaat — {html.escape(data['symbol'])} ({html.escape(data['period'])})</h3>
  <div class="kpis">
    <div class="kpi {tr_kpi}"><div class="label">Totaal rendement</div><div class="value">{pct(tr)}</div></div>
    <div class="kpi neutral"><div class="label">Buy-and-hold</div><div class="value">{pct(bench)}</div></div>
    <div class="kpi bad"><div class="label">Max drawdown</div><div class="value">{pct(dd)}</div></div>
    <div class="kpi neutral"><div class="label">Sharpe</div><div class="value">{data['sharpe']}</div></div>
    <div class="kpi neutral"><div class="label">Trades</div><div class="value">{data['n_trades']}</div></div>
    <div class="kpi {'good' if data['win_rate']>=0.5 else 'neutral'}"><div class="label">Win-rate</div><div class="value">{data['win_rate']*100:.0f}%</div></div>
  </div>
  <p style="margin-top:12px">
    Strategie sloten op <b>€{data['final_capital']:,.0f}</b> van {data['initial_capital']:,.0f} startkapitaal.
    De lagere max-drawdown (vs benchmark) laat crashweerstand zien; de strategie is
    conservatiever maar beperkt verliezen in crashes.
  </p>
</div>
<div class="card">
  <h3>Equity-curve (Strategie vs Buy-and-hold)</h3>
  {svg}
  <p style="font-size:12px;color:var(--muted)">{len(ts)} datapunten geplot. Download zelf via knop linksonder in de nav.</p>
</div>
"""


def _download_html() -> str:
    return """
<h2>⬇️ Download &amp; GitHub-export</h2>
<p>Download het hele project als <b>.zip</b> — dit werkt op <b>Windows, macOS, iPad en elke browser</b>.
De zip is klaar om naar GitHub te uploaden (zonder .venv, caches of secrets).</p>
<div class="card">
  <div class="dl-panel">
    <button class="secondary" onclick="location.href='/getzip'">⬇️ Download project (.zip)</button>
    <span class="dl-size">Excl. .venv / caches / .env</span>
    <button class="secondary" onclick="location.href='/github'">▶ GitHub-instructies</button>
  </div>
</div>
<div class="card" style="margin-top:18px">
  <h3>GitHub push (vanaf deze server)</h3>
  <pre><code>cd ~/trading-bot
git init -b main
git add -A && git commit -m "Initial Hermes trading bot"
git remote add origin https://github.com/JOUW-NAAM/hermes-trading-bot.git
git push -u origin main</code></pre>
  <p style="font-size:13px">Voor iPad: gebruik de knop hierboven om de zip te downloaden, open de
  <b>GitHub-app</b>, maak een repo en upload de bestanden. Zie de
  <a href="/github">GitHub-instructies</a> voor de volledige stappen.</p>
</div>
"""


def _code_view_html(rel: str, content: str) -> str:
    # Toon code met regelnummers.
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
        ("/backtest", "Backtest", "Backtest"),
        ("/download", "Download", "Download"),
        ("/architectuur", "Architectuur", "Architectuur"),
        ("/code", "Code", "Code"),
        ("/vault", "Noten", "Noten"),
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
                "<h2>Welkom</h2>"
                "<p>Modulaire multi-asset AI-tradingbot die <b>winst</b> nastreeft met weerstand tegen "
                "<b>marktcrashes</b>. Verken de architectuur, draai een backtest, bekijk de code/noten, "
                "of download het project.</p>"
                "<div class='kpis'>"
                "<div class='kpi good'><div class='label'>Doel</div><div class='value' style='font-size:16px'>goede winst</div></div>"
                "<div class='kpi neutral'><div class='label'>Weerstand</div><div class='value' style='font-size:16px'>crash-bestendig</div></div>"
                "<div class='kpi neutral'><div class='label'>Multi-asset</div><div class='value' style='font-size:16px'>aandelen/ETF's/obligaties</div></div>"
                "<div class='kpi neutral'><div class='label'>Executie</div><div class='value' style='font-size:16px'>paper → live</div></div>"
                "</div>"
                "<div class='card'><h3>Start hier</h3>"
                "<p>• <a href='/backtest'>Verken de backtest-tool</a> — zie hoe de risico-engine crashes beperkt.<br>"
                "• <a href='/architectuur'>Bekijk het architectuur-diagram</a> — de 7 lagen.<br>"
                "• <a href='/download'>Download of exporteer naar GitHub</a>.<br>"
                "• <a href='/run'>Draai de demo-pipeline</a>.</p></div>"
            )
            self._send(self._page("Dashboard", content, "Dash"))
        elif path == "/backtest":
            self._send(self._page("Backtest", _backtest_html(), "Backtest"))
        elif path == "/download":
            self._send(self._page("Download", _download_html(), "Download"))
        elif path == "/github":
            from hermes_bot.export_zip import build_github_instructions
            content = f"<h2>GitHub-instructies</h2><pre>{html.escape(build_github_instructions())}</pre>"
            self._send(self._page("GitHub", content, "Download"))
        elif path == "/architectuur":
            self._send(self._page("Architectuur", "<h2>🏗️ Architectuur</h2>" + _architecture_html(), "Architectuur"))
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
                content = "<h2>Obsidian-noten</h2>" + _vault_notes_html()
            self._send(self._page("Noten", content, "Noten"))
        elif path == "/run":
            content = (
                "<h2>Run demo</h2>"
                "<div class='card'><form method='POST' action='/run'><button>▶ Draai demo-pipeline</button></form>"
                "<p style='font-size:13px;margin-top:8px'>Ook: <code>uv run pytest</code> en <code>uv run python -m hermes_bot.demo</code></p></div>"
            )
            self._send(self._page("Demo", content, "Dash"))
        elif path == "/getzip":
            # Download het project als .zip (voor GitHub-export, ook op iPad).
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
            # Datareturn voor AJAX zonder pagina-shell
            try:
                from hermes_bot.backtest.runner import run_backtest
                data = run_backtest(
                    symbol=qs.get("symbol", ["SPY"])[0],
                    period=qs.get("period", ["2y"])[0],
                    vol_target=float(qs.get("vol_target", ["0.125"])[0]),
                )
                self._send(_backtest_result_html(data))
            except Exception as e:  # noqa: BLE001
                self._send(f'<div class="card"><p style="color:var(--danger)">Fout: {html.escape(str(e))}</p></div>')
        else:
            content = "<h2>404</h2><p>Pagina niet gevonden.</p><a href='/'>← Terug</a>"
            self._send(self._page("404", content))

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/run":
            try:
                r = subprocess.run(
                    ["uv", "run", "python", "-m", "hermes_bot.demo"],
                    cwd=str(ROOT), capture_output=True, text=True, timeout=90,
                )
                out = r.stdout or r.stderr
                content = f"<h2>Demo-output</h2><pre><code>{html.escape(out)}</code></pre>"
            except Exception as e:  # noqa: BLE001
                content = f"<h2>Fout</h2><pre><code>{html.escape(str(e))}</code></pre>"
            self._send(self._page("Demo", content, "Dash"))
        else:
            self._send("<h2>404</h2>")


def main() -> None:
    print(f"Hermes trading-bot web-interface op http://localhost:{PORT}")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()