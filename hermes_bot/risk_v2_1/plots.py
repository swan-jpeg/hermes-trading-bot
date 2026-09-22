"""Visualisaties voor Risk Engine v2.1 — equity, drawdown, exposure, scores.

Genereert HTML-plots naar var/forensic_v21/plots/.
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent.parent / "var" / "forensic_v21" / "plots"


def _svg_line(series, color, w=900, h=300):
    if not series:
        return ""
    mx, mn = max(series), min(series)
    rng = (mx - mn) or 1.0
    n = len(series)
    pts = []
    for i, v in enumerate(series):
        x = 10 + i * (w - 20) / max(1, n - 1)
        y = h - 10 - (v - mn) / rng * (h - 20)
        pts.append(f"{x:.1f},{y:.1f}")
    return (f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" '
            f'stroke-width="2" stroke-linejoin="round"/>')


def _svg_area(series, color, w=900, h=300):
    if not series:
        return ""
    n = len(series)
    pts = []
    for i, v in enumerate(series):
        x = 10 + i * (w - 20) / max(1, n - 1)
        y = h - 10 - v * (h - 20)
        pts.append(f"{x:.1f},{y:.1f}")
    base = f"10,{h-10} {w-10},{h-10}"
    return (f'<polygon points="{base} {" ".join(pts)}" fill="{color}" '
            f'fill-opacity="0.3" stroke="{color}" stroke-width="1.5"/>')


def _page(title, body):
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>{title}</title>
<style>
body{{font-family:-apple-system,system-ui,sans-serif;background:#f5f5f7;color:#1d1d1f;margin:0;padding:24px}}
h1{{font-size:22px;letter-spacing:-0.02em}} h2{{font-size:16px;margin-top:28px}}
.card{{background:#fff;border-radius:14px;box-shadow:0 4px 20px rgba(0,0,0,.06);
  padding:18px;margin-bottom:18px}}
svg{{width:100%;height:auto}}
</style></head><body>{body}</body></html>"""


def generate_plots(scenario: str = "fast_crash") -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    log_path = OUT.parent / "scenarios" / f"v21_{scenario}_log.json"
    if not log_path.exists():
        return
    log = json.loads(log_path.read_text())
    equity = [e["portfolio_value"] for e in log]
    exposure = [e["effective_exposure"] for e in log]
    risk = [e["risk_score"] for e in log]
    opp = [e["opp_score"] for e in log]
    vol = [e["vol"] for e in log]
    dd = [e["drawdown"] for e in log]

    body = f"""
<h1>Risk Engine v2.1 — {scenario}</h1>
<div class="card"><h2>Equity-curve</h2>
<svg viewBox="0 0 900 300">{_svg_line(equity, '#0a84ff')}</svg></div>
<div class="card"><h2>Drawdown</h2>
<svg viewBox="0 0 900 300">{_svg_area([abs(d) for d in dd], '#ff453a')}</svg></div>
<div class="card"><h2>Exposure over tijd</h2>
<svg viewBox="0 0 900 300">{_svg_area(exposure, '#30d158')}</svg></div>
<div class="card"><h2>Risk Environment Score (0-100)</h2>
<svg viewBox="0 0 900 300">{_svg_line(risk, '#ff9f0a')}</svg></div>
<div class="card"><h2>Opportunity Score (0-100)</h2>
<svg viewBox="0 0 900 300">{_svg_line(opp, '#bf5af2')}</svg></div>
<div class="card"><h2>Volatiliteit</h2>
<svg viewBox="0 0 900 300">{_svg_line(vol, '#ff375f')}</svg></div>
"""
    (OUT / f"{scenario}.html").write_text(_page(f"v2.1 {scenario}", body))


def generate_all() -> None:
    for sc in ["bull", "bear", "fast_crash", "flash_crash", "v_shape",
               "slow_drawdown", "vol_spike", "multi_shock"]:
        generate_plots(sc)


if __name__ == "__main__":
    generate_all()
    print(f"v2.1 plots geschreven naar {OUT}")
