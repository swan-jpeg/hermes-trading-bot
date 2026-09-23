"""OOS visualizations — equity, drawdown, exposure, regime for the test period.

Genereert HTML-plots naar var/oos_v1/plots/.
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent.parent / "var" / "oos_v1" / "plots"


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
.legend{{display:flex;gap:16px;font-size:13px;color:#6e6e73;margin-top:6px}}
.swatch{{width:14px;height:4px;border-radius:2px;display:inline-block;margin-right:6px}}
</style></head><body>{body}</body></html>"""


def generate_plots(sym: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    log_path = OUT.parent / f"{sym}_v2_daily_log.json"
    if not log_path.exists():
        return
    log = json.loads(log_path.read_text())
    equity = [e["portfolio_value"] for e in log]
    exposure = [e["final_exposure"] for e in log]
    vol = [e["volatility"] for e in log]
    dd = [e["drawdown"] for e in log]
    regime = [e["regime"] for e in log]
    risk = [e["vol_multiplier"] * e["drawdown_multiplier"] * e["regime_multiplier"]
            for e in log]

    # Regime as numeric (for plotting).
    regime_map = {"normal": 1, "elevated": 0.8, "stressed": 0.5, "crash": 0.2, "recovery": 0.6}
    regime_num = [regime_map.get(r, 1) for r in regime]

    body = f"""
<h1>Risk Engine v2 — OOS {sym} (2024-2026)</h1>
<div class="card"><h2>Equity-curve</h2>
<svg viewBox="0 0 900 300">{_svg_line(equity, '#0a84ff')}</svg></div>
<div class="card"><h2>Drawdown</h2>
<svg viewBox="0 0 900 300">{_svg_area([abs(d) for d in dd], '#ff453a')}</svg></div>
<div class="card"><h2>Exposure over tijd</h2>
<svg viewBox="0 0 900 300">{_svg_area(exposure, '#30d158')}</svg></div>
<div class="card"><h2>Volatiliteit</h2>
<svg viewBox="0 0 900 300">{_svg_line(vol, '#ff9f0a')}</svg></div>
<div class="card"><h2>Risk score (product van multipliers)</h2>
<svg viewBox="0 0 900 300">{_svg_line(risk, '#bf5af2')}</svg></div>
<div class="card"><h2>Regime (normal=1, elevated=0.8, stressed=0.5, crash=0.2, recovery=0.6)</h2>
<svg viewBox="0 0 900 300">{_svg_line(regime_num, '#ff375f')}</svg></div>
"""
    (OUT / f"{sym}.html").write_text(_page(f"OOS {sym}", body))


def generate_all() -> None:
    for sym in ["SPY", "QQQ", "IWM"]:
        generate_plots(sym)


if __name__ == "__main__":
    generate_all()
    print(f"OOS plots geschreven naar {OUT}")
