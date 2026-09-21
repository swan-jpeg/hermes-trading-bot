"""Web-interface voor de tradingbot (poort 9124).

Toont: code/mappen, Obsidian-noten (rendered markdown), en laat de
demo/tests runnen. Zelfstandig met stdlib + markdown-it-py.
Gebruik: `uv run python -m hermes_bot.webui`  (of via systemd)
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


def _tree_html(path: Path, base: Path) -> str:
    """Recursieve file-tree als HTML (alleen .py/.md/.yaml/.toml/.txt)."""
    out: list[str] = []
    try:
        entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    except OSError:
        return ""
    for p in entries:
        if p.name.startswith(".") or p.name in ("__pycache__", ".venv", "node_modules"):
            continue
        rel = p.relative_to(base)
        if p.is_dir():
            out.append(
                f'<details><summary class="dir">📁 {html.escape(p.name)}</summary>'
                f"{_tree_html(p, base)}</details>"
            )
        elif p.suffix in (".py", ".md", ".yaml", ".yml", ".toml", ".txt", ".json"):
            out.append(
                f'<div class="file"><a href="/code?path={urllib.parse.quote(str(rel))}">'
                f"📄 {html.escape(p.name)}</a></div>"
            )
    return "".join(out)


def _vault_notes_html() -> str:
    """Lijst Obsidian-noten in TradingBot/."""
    tb = VAULT / "TradingBot"
    out: list[str] = []
    if tb.is_dir():
        for p in sorted(tb.glob("*.md")):
            out.append(
                f'<div class="file"><a href="/vault?note={urllib.parse.quote(p.name)}">'
                f"📝 {html.escape(p.stem)}</a></div>"
            )
    return "".join(out)


PAGE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Hermes Trading Bot</title>
<style>
body{{font-family:system-ui,sans-serif;margin:0;background:#0f1117;color:#e6e6e6}}
header{{background:#161a24;padding:14px 24px;border-bottom:1px solid #2a2f3a}}
header h1{{margin:0;font-size:20px}}
nav a{{color:#7aa2f7;margin-right:16px;text-decoration:none;font-weight:600}}
main{{display:flex;gap:0;min-height:calc(100vh - 60px)}}
.side{{width:340px;background:#12151d;padding:16px;border-right:1px solid #2a2f3a;overflow:auto}}
.content{{flex:1;padding:24px;overflow:auto}}
pre{{background:#0b0d12;padding:14px;border-radius:8px;overflow:auto;font-size:13px}}
code{{font-family:ui-monospace,monospace}}
.dir{{cursor:pointer;font-weight:600;color:#9ece6a}}
.file{{padding:2px 0}}
.file a{{color:#c0caf5;text-decoration:none}}
.file a:hover{{color:#7aa2f7}}
button{{background:#7aa2f7;color:#0f1117;border:none;padding:8px 16px;border-radius:6px;font-weight:700;cursor:pointer}}  # noqa: E501
button:hover{{background:#9aa5ce}}
h2{{border-bottom:1px solid #2a2f3a;padding-bottom:6px}}
table{{border-collapse:collapse;width:100%}}
td,th{{border:1px solid #2a2f3a;padding:6px 10px;text-align:left}}
</style></head><body>
<header><h1>🤖 Hermes Trading Bot</h1>
<nav>
<a href="/">Dashboard</a>
<a href="/code">Code</a>
<a href="/vault">Obsidian-noten</a>
<a href="/run">Run demo</a>
</nav></header>
<main>
<div class="side">
<h3>📁 Project</h3>
{tree}
<h3>📝 Obsidian (TradingBot)</h3>
{vault}
</div>
<div class="content">
{content}
</div>
</main>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, body: str, ctype: str = "text/html; charset=utf-8") -> None:
        data = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _page(self, content: str) -> str:
        return PAGE.format(
            tree=_tree_html(ROOT / "hermes_bot", ROOT),
            vault=_vault_notes_html(),
            content=content,
        )

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(parsed.query)
        path = parsed.path

        if path == "/":
            content = (
                "<h2>Dashboard</h2>"
                "<p>Modulaire multi-asset AI-trading-bot. Kies links een module of "
                "Obsidian-noot, of <a href='/run'>run de demo</a>.</p>"
                "<h3>Architectuurketen</h3><pre>"
                "data → fusie → RL → risico → executie (paper)\n"
                "AI agent → Monte Carlo → risico-engine → output actie</pre>"
            )
        elif path == "/code":
            rel = qs.get("path", [""])[0]
            target = (ROOT / rel).resolve()
            if target.is_file() and ROOT in target.parents:
                content = f"<h2>📄 {html.escape(rel)}</h2><pre>{html.escape(target.read_text())}</pre>"  # noqa: E501
            else:
                content = "<h2>Code</h2>" + _tree_html(ROOT / "hermes_bot", ROOT)
        elif path == "/vault":
            note = qs.get("note", [""])[0]
            target = VAULT / "TradingBot" / note
            if target.is_file():
                content = f"<h2>📝 {html.escape(target.stem)}</h2>" + _render_md(target.read_text())
            else:
                content = "<h2>Obsidian-noten</h2>" + _vault_notes_html()
        elif path == "/run":
            content = (
                "<h2>Run demo</h2>"
                "<form method='POST' action='/run'><button>▶ Draai demo-pipeline</button></form>"
                "<p><small>Ook: <code>uv run pytest</code> en <code>uv run python -m hermes_bot.demo</code></small></p>"  # noqa: E501
            )
        else:
            content = "<h2>404</h2>"
        self._send(self._page(content))

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/run":
            try:
                r = subprocess.run(
                    ["uv", "run", "python", "-m", "hermes_bot.demo"],
                    cwd=str(ROOT), capture_output=True, text=True, timeout=60,
                )
                out = r.stdout or r.stderr
                content = f"<h2>Demo-output</h2><pre>{html.escape(out)}</pre>"
            except Exception as e:  # noqa: BLE001
                content = f"<h2>Fout</h2><pre>{html.escape(str(e))}</pre>"
            self._send(self._page(content))
        else:
            self._send("<h2>404</h2>")


def main() -> None:
    print(f"Hermes trading-bot web-interface op http://localhost:{PORT}")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
