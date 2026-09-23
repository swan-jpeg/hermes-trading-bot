"""Visualizations for Risk Engine v2 — equity, drawdown, exposure, risk score.

Genereert HTML-plots (zelfstandig, geen matplotlib nodig) naar var/forensic_v2/plots/.
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent.parent / "var" / "forensic_v2" / "plots"


def _svg_line(ts, series, color, w=900, h=300, label=""):
    """SVG-lijndiagram."""
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


def _svg_area(ts, series, color, w=900, h=300):
    """SVG area chart (for exposure 0..1)."""
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
.legend{{display:flex;gap:16px;font-size:13px;color:#6e6e73;margin-top:6px}}
.swatch{{width:14px;height:4px;border-radius:2px;display:inline-block;margin-right:6px}}
</style></head><body>{body}</body></html>"""


def generate_plots(scenario: str = "fast_crash") -> None:
    """Generate plots for a scenario (v2 log)."""
    OUT.mkdir(parents=True, exist_ok=True)
    log_path = OUT.parent / "scenarios" / f"v2_{scenario}_log.json"
    if not log_path.exists():
        return
    log = json.loads(log_path.read_text())
    ts = [e["date"] for e in log]
    equity = [e["portfolio_value"] for e in log]
    exposure = [e["final_exposure"] for e in log]
    vol = [e["volatility"] for e in log]
    risk = [e["vol_multiplier"] * e["drawdown_multiplier"] * e["regime_multiplier"]
            for e in log]

    # Equity + drawdown.
    peak = []
    p = 0
    for v in equity:
        p = max(p, v)
        peak.append(p)
    dd_series = [v / p - 1 for v, p in zip(equity, peak, strict=False)]

    body = f"""
<h1>Risk Engine v2 — {scenario}</h1>
<div class="card"><h2>Equity-curve</h2>
<svg viewBox="0 0 900 300">{_svg_line(ts, equity, '#0a84ff')}</svg></div>
<div class="card"><h2>Drawdown</h2>
<svg viewBox="0 0 900 300">{_svg_area(ts, [abs(d) for d in dd_series], '#ff453a')}</svg></div>
<div class="card"><h2>Exposure over tijd</h2>
<svg viewBox="0 0 900 300">{_svg_area(ts, exposure, '#30d158')}</svg></div>
<div class="card"><h2>Volatiliteit</h2>
<svg viewBox="0 0 900 300">{_svg_line(ts, vol, '#ff9f0a')}</svg></div>
<div class="card"><h2>Risk score (product van multipliers)</h2>
<svg viewBox="0 0 900 300">{_svg_line(ts, risk, '#bf5af2')}</svg></div>
"""
    (OUT / f"{scenario}.html").write_text(_page(f"v2 {scenario}", body))


def generate_all() -> None:
    for sc in ["bull", "bear", "fast_crash", "flash_crash", "v_shape",
               "slow_drawdown", "multi_shock"]:
        generate_plots(sc)


if __name__ == "__main__":
    generate_all()
    print(f"Plots geschreven naar {OUT}")
