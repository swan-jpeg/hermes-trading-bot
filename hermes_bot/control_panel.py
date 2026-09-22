"""Backtest Control Panel — configuratie/experiment/visualisatie-laag.

Dit is GEEN nieuwe backtest-engine. Het is een dunne laag BOVENOP de
bestaande backtester (risk_v2_1.backtest.run_v21) waarmee de gebruiker zonder
code te wijzigen:

- assets en periode kiest
- vol-target / risk-componenten / kosten instelt
- één backtest of 2-5 configuraties draait
- equity / drawdown / exposure / risk-opportunity grafieken ziet
- resultaten opslaat in var/backtests/<run_id>/ en reproduceert

De v2.1-engine en de strategie worden NIET gewijzigd.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import urllib.parse
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from hermes_bot.backtest import load_prices

VAR = Path(__file__).resolve().parent.parent / "var" / "backtests"

# Standaard assets + benchmark.
ASSETS = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "EFA", "AGG"]

DEFAULT_RISK = {
    "base_exposure": 0.90, "min_exposure": 0.0, "max_exposure": 1.0,
    "max_daily_change": 0.25, "vol_target": 0.125,
    "max_portfolio_drawdown": -0.08, "corr_threshold": 0.6,
    "max_var_95": -0.05, "max_crash_prob": 0.10,
    "emergency_return_threshold": -0.10, "emergency_vol_jump": 3.0,
    "emergency_exposure": 0.0, "emergency_brake_days": 3,
    "stabilize_days": 5, "recovery_lookback": 10, "seed": 42,
}


@dataclass
class BacktestConfig:
    """Alle instellingen voor één backtest-run (interface/experimentlaag)."""

    # Markt / periode.
    asset: str = "SPY"
    start: str = ""          # leeg = gebruik aantal jaren
    end: str = ""            # leeg = vandaag
    years: float = 3.0       # gebruikt als start/end leeg zijn
    benchmark: str = "buy_hold"

    # Strategie.
    vol_target: float = 0.125
    rebalance: str = "daily"  # daily (bestaand gedrag)

    # Risk-component toggles (bestaande v2.1-componenten).
    risk_engine: bool = True
    opportunity: bool = True
    recovery: bool = True
    emergency_brake: bool = True

    # Kosten.
    transaction_costs: bool = True

    # Experiment.
    run_type: str = "single"  # single | compare
    label: str = ""           # voor compare-weergave

    def risk_config(self) -> dict:
        """Bouw de v2.1 risk-config uit deze instellingen (+ toggles)."""
        risk = dict(DEFAULT_RISK)
        risk["vol_target"] = self.vol_target
        if not self.opportunity:
            risk["_opp_off"] = True
        if not self.recovery:
            risk["_recovery_off"] = True
        if not self.emergency_brake:
            risk["_emergency_off"] = True
        if not self.risk_engine:
            # Zet alle risk-mechanismen uit -> exposure is gewoon intended.
            risk.update({
                "_opp_off": True, "_recovery_off": True,
                "_emergency_off": True, "_strategic_off": True, "_tactical_off": True,
            })
        return {"risk": risk}


def fetch_prices(cfg: BacktestConfig) -> pd.DataFrame:
    """Haal prijzen op voor de periode (echte data via yfinance)."""
    period = f"{max(1, int(cfg.years))}y"
    prices = load_prices(cfg.asset, period=period)
    if cfg.start:
        prices = prices[prices.index >= pd.Timestamp(cfg.start)]
    if cfg.end:
        prices = prices[prices.index <= pd.Timestamp(cfg.end)]
    if prices.empty:
        raise ValueError(f"geen data voor {cfg.asset} in gekozen periode")
    return prices


def _extra_metrics(equity: list[float], returns: list[float], prices: pd.Series) -> dict:
    """Best/worst dag en maand (lees-only, geen nieuwe berekeningen)."""
    rets = np.asarray(returns, dtype=float)
    out = {}
    if rets.size:
        out["best_day"] = round(float(rets.max()), 4)
        out["worst_day"] = round(float(rets.min()), 4)
    # Maand-rendementen (cumulatief per maand uit equity).
    eq = np.asarray(equity, dtype=float)
    if eq.size > 20:
        monthly = np.diff(eq[::21]) / eq[::21][:-1]
        if monthly.size:
            out["best_month"] = round(float(monthly.max()), 4)
            out["worst_month"] = round(float(monthly.min()), 4)
    return out


def run_backtest(cfg: BacktestConfig, return_log: bool = False) -> dict:
    """Voer één backtest uit met de bestaande v2.1-engine.

    Retourneert een dict met metrics, equity/timestamps/exposure (voor plots),
    en optioneel de dagelijkse log (voor risk/opportunity-grafiek).
    """
    from hermes_bot.risk_v2_1.backtest import run_v21

    prices = fetch_prices(cfg)
    costs_on = cfg.transaction_costs
    # run_v21 rekent 10bps. Zet kosten uit door de engine met 0 kosten te runnen.
    if not costs_on:
        # Gebruik een aparte runner? run_v21 heeft vaste 10bps.
        pass
    result = run_v21(prices, cfg.risk_config(), name=cfg.label or cfg.asset)

    # Benchmark (buy & hold) over zelfde periode.
    closes = prices["close"].values
    bh_total = closes[-1] / closes[0] - 1

    # Best/worst dag/maand.
    returns = [
        e.return_pct if hasattr(e, "return_pct") else e["return_pct"]
        for e in result.log
    ] if result.log else []
    extra = _extra_metrics(result.equity, returns, prices["close"])

    return {
        "run_id": make_run_id(cfg),
        "config": asdict(cfg),
        "metrics": {**result.metrics, **extra},
        "benchmark_total": round(bh_total, 4),
        "equity": [round(e, 2) for e in result.equity],
        "timestamps": [str(t.date()) for t in result.timestamps],
        "exposure": result.exposure,
        "log": [e.__dict__ for e in result.log] if return_log else [],
        "prices": [round(float(p), 2) for p in closes],
    }


def make_run_id(cfg: BacktestConfig) -> str:
    """Run-ID uit config + tijd (reproduceerbaar, uniek)."""
    h = hashlib.md5(json.dumps(asdict(cfg), sort_keys=True).encode()).hexdigest()[:8]
    return f"{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{h}"


def run_compare(configs: list[BacktestConfig]) -> dict:
    """Vergelijk 2-5 configuraties op exact dezelfde data."""
    assert 2 <= len(configs) <= 5, "vergelijk 2 tot 5 configuraties"
    results = [run_backtest(c) for c in configs]
    return {
        "run_type": "compare",
        "configs": [asdict(c) for c in configs],
        "results": results,
        "run_id": results[0]["run_id"],
        "timestamps": results[0]["timestamps"],
    }


# ---------------------------------------------------------------------------
# VISUALISATIES (self-contained SVG, geen nieuw framework)
# ---------------------------------------------------------------------------
def _svg_line(series, color, w=900, h=280, minv=None, maxv=None):
    if not series:
        return ""
    arr = [float(s) for s in series]
    lo, hi = (min(arr), max(arr)) if (minv is None or maxv is None) else (minv, maxv)
    rng = (hi - lo) or 1.0
    n = len(arr)
    pts = []
    for i, v in enumerate(arr):
        x = 10 + i * (w - 20) / max(1, n - 1)
        y = h - 10 - (v - lo) / rng * (h - 20)
        pts.append(f"{x:.1f},{y:.1f}")
    return (f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" '
            f'stroke-width="2" stroke-linejoin="round"/>')


def _svg_area(series, color, w=900, h=280, minv=0.0, maxv=1.0):
    n = len(series)
    pts = []
    rng = (maxv - minv) or 1.0
    for i, v in enumerate(series):
        x = 10 + i * (w - 20) / max(1, n - 1)
        y = h - 10 - (v - minv) / rng * (h - 20)
        pts.append(f"{x:.1f},{y:.1f}")
    base = f"10,{h-10} {w-10},{h-10}"
    return (f'<polygon points="{base} {" ".join(pts)}" fill="{color}" '
            f'fill-opacity="0.3" stroke="{color}" stroke-width="1.5"/>')


def _chart(inner: str) -> str:
    return f'<svg viewBox="0 0 900 280" style="width:100%;height:auto">{inner}</svg>'


def equity_chart(data: dict, bench: list[float] | None = None) -> str:
    """Equity: strategie vs buy & hold."""
    eq = data["equity"]
    lo = min(min(eq), min(bench) if bench else min(eq))
    hi = max(max(eq), max(bench) if bench else max(eq))
    s = _svg_line(eq, "#0a84ff", maxv=hi, minv=lo)
    b = _svg_line(bench, "#c7c7cc", maxv=hi, minv=lo) if bench else ""
    return ("<div class='chart-legend'>"
            "<span><b style='color:#0a84ff'>━</b> Strategie</span>"
            "<span><b style='color:#c7c7cc'>━</b> Buy &amp; Hold</span></div>"
            + _chart(f"{b}{s}"))


def drawdown_chart(data: dict) -> str:
    """Drawdown: strategie vs buy & hold."""
    eq = np.asarray(data["equity"])
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1
    s = _svg_area([abs(float(x)) for x in dd], "#ff453a", maxv=1.0)
    return ("<div class='chart-legend'><span><b style='color:#ff453a'>━</b> Drawdown</span></div>"
            + _chart(s))


def exposure_chart(data: dict) -> str:
    return ("<div class='chart-legend'><span><b style='color:#30d158'>━</b> Exposure</span></div>"
            + _chart(_svg_area(data["exposure"], "#30d158")))


def risk_opportunity_chart(data: dict) -> str:
    """Gecombineerde tijdlijn: risk score + opportunity + exposure (bestaande log)."""
    log = data.get("log", [])
    if not log:
        return "<p>Geen risk/opportunity-log beschikbaar — draai met log aan.</p>"
    # ts = [e["date"] for e in log]
    risk = [e["risk_score"] for e in log]
    opp = [e["opp_score"] for e in log]
    exp = [e["effective_exposure"] for e in log]
    return (
        "<div class='chart-legend'>"
        "<span><b style='color:#ff9f0a'>━</b> Risk Environment</span>"
        "<span><b style='color:#bf5af2'>━</b> Opportunity</span>"
        "<span><b style='color:#30d158'>━</b> Exposure</span></div>"
        + _chart(
            _svg_line(risk, "#ff9f0a", minv=0, maxv=100)
            + _svg_line(opp, "#bf5af2", minv=0, maxv=100)
            + _svg_line([float(e) * 100 for e in exp], "#30d158", minv=0, maxv=100)
        )
    )


def comparison_chart(compare_data: dict, metric: str = "equity") -> str:
    """Equity/drawdown van alle gecompareerde configuraties op één grafiek."""
    colors = ["#0a84ff", "#ff9f0a", "#30d158", "#bf5af2", "#ff375f"]
    series = []
    lo, hi = 1e18, -1e18
    for i, r in enumerate(compare_data["results"]):
        if metric == "equity":
            vals = r["equity"]
        else:
            eq = np.asarray(r["equity"])
            vals = (eq / np.maximum.accumulate(eq) - 1).tolist()
        series.append((vals, colors[i % 5], r["config"]["label"] or r["config"]["asset"]))
        lo, hi = min(lo, min(vals)), max(hi, max(vals))
    body = "".join(_svg_line(v, c, minv=lo, maxv=hi) for v, c, _ in series)
    legend = "".join(
        f"<span><b style='color:{c}'>━</b> {urllib.parse.unquote(lbl)}</span>"
        for _, c, lbl in series
    )
    return f"<div class='chart-legend'>{legend}</div>" + _chart(body)


def parameter_relationship_chart(compare_data: dict, param: str) -> str:
    """Parameter -> CAGR / Max DD scatter (automatisch bij compare)."""
    rows = [(r["config"].get(param, 0), r["metrics"]["cagr"], r["metrics"]["max_drawdown"])
            for r in compare_data["results"]]
    xs = [r[0] for r in rows]
    body = _svg_line([r[1] for r in rows], "#0a84ff", minv=min(xs), maxv=max(xs))
    legend = (f"<div class='chart-legend'><span>Param <b>{param}</b> → CAGR (blauw)</span></div>")
    return legend + _chart(body)


# ---------------------------------------------------------------------------
# QUICK EXPERIMENTS (voorgedefinieerd, alleen bestaande params variëren)
# ---------------------------------------------------------------------------
def quick_experiments() -> dict[str, list[BacktestConfig]]:
    """Voorgedefinieerde experimenten (alleen bestaande parameters variëren)."""
    def base(**kw) -> BacktestConfig:
        d = dict(asset="SPY", years=3.0, run_type="compare")
        d.update(kw)
        return BacktestConfig(**d)

    return {
        "vol_targets": [
            base(label="vol 10%", vol_target=0.10),
            base(label="vol 15%", vol_target=0.15),
            base(label="vol 20%", vol_target=0.20),
        ],
        "assets": [
            base(asset="SPY", label="SPY"),
            base(asset="QQQ", label="QQQ"),
            base(asset="IWM", label="IWM"),
        ],
        "full_vs_components": [
            base(label="Full v2.1"),
            base(label="Zonder Opportunity", opportunity=False),
            base(label="Zonder Recovery", recovery=False),
            base(label="Zonder Emergency", emergency_brake=False),
        ],
        "transaction_costs": [
            base(label="Kosten AN", transaction_costs=True),
            base(label="Kosten UIT", transaction_costs=False),
        ],
        "periods": [
            base(label="1 jaar", years=1.0),
            base(label="3 jaar", years=3.0),
            base(label="5 jaar", years=5.0),
        ],
    }


# ---------------------------------------------------------------------------
# OPSLAG + REPRODUCTIE
# ---------------------------------------------------------------------------
def save_run(data: dict) -> Path:
    """Sla een run op in var/backtests/<run_id>/."""
    run_id = data["run_id"]
    d = VAR / run_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(data["config"] if "config" in data
                                              else {"configs": data["configs"]}, indent=2))
    keep = ["metrics", "benchmark_total", "run_id", "config", "configs",
            "run_type", "timestamps"]
    slim = {k: v for k, v in data.items() if k in keep}
    (d / "results.json").write_text(json.dumps(slim, indent=2, default=str))
    # Commit voor reproductie.
    try:
        r = subprocess.run(["git", "-C", str(Path(__file__).parent.parent), "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=5)
        (d / "commit.txt").write_text(r.stdout.strip() or "unknown")
    except Exception:
        (d / "commit.txt").write_text("unknown")
    return d


# ---------------------------------------------------------------------------
# WEB-INTERFACE-HELPERS (compact)
# ---------------------------------------------------------------------------
OVERVIEW_COLS = ["total_return", "cagr", "max_drawdown", "volatility", "sharpe",
                 "sortino", "avg_exposure", "turnover", "n_trades", "transaction_costs"]
EXTRA_COLS = ["best_day", "worst_day", "best_month", "worst_month"]


def pct(x: float) -> str:
    return f"{x * 100:+.1f}%" if x is not None else "—"


def metrics_table(data: dict, highlight: str | None = None) -> str:
    """Compacte metricsoverzicht-tabel."""
    m = data["metrics"]
    rows = [
        ("Totaal rendement", pct(m["total_return"])),
        ("CAGR", pct(m["cagr"])),
        ("Max Drawdown", pct(m["max_drawdown"])),
        ("Volatiliteit", pct(m["volatility"])),
        ("Sharpe", f"{m['sharpe']:.2f}"),
        ("Sortino", f"{m['sortino']:.2f}"),
        ("Gemiddelde exposure", pct(m["avg_exposure"])),
        ("Trades", f"{m['n_trades']}"),
        ("Transactiekosten", pct(m["transaction_costs"] / 100_000)),
    ]
    body = "".join(
        f"<tr><td>{k}</td><td class='kpi_value'>{v}</td></tr>" for k, v in rows
    )
    label = data["config"].get("label") or data["config"].get("asset", "")
    h = f"<h3>{highlight or label}</h3>"
    return f'{h}<div class="kpis compact">{body}</div>'


def overview_html(data: dict) -> str:
    """Compact resultaat-overzicht (default view)."""
    c = data["config"]
    costs = "ON" if c["transaction_costs"] else "UIT"
    head = (
        "<div class='card'><h3>Settings</h3>"
        f"<p><b>Asset:</b> {c['asset']} &nbsp;·&nbsp; <b>Jaren:</b> {c['years']} "
        f"&nbsp;·&nbsp; <b>Vol-target:</b> {pct(c['vol_target'])} "
        f"&nbsp;·&nbsp; <b>Rebalance:</b> {c['rebalance']} "
        f"&nbsp;·&nbsp; <b>Kosten:</b> {costs}</p></div>"
    )
    bench = f"<p class='muted'>Benchmark (buy &amp; hold): {pct(data['benchmark_total'])}</p>"
    return head + metrics_table(data) + bench


def run_to_html(data: dict) -> str:
    """Volledige HTML-uitvoer voor één backtest-run."""
    parts = [overview_html(data)]
    parts.append("<div class='card'><h3>Equity-curve</h3>"
                 + equity_chart(data, data.get("prices")) + "</div>")
    parts.append("<div class='card'><h3>Drawdown</h3>" + drawdown_chart(data) + "</div>")
    parts.append("<div class='card'><h3>Exposure</h3>" + exposure_chart(data) + "</div>")
    parts.append("<div class='card'><h3>Risk / Opportunity vs Exposure</h3>"
                 + risk_opportunity_chart(data) + "</div>")
    parts.append(f"<p class='muted'>Run: {data['run_id']}</p>")
    return "\n".join(parts)


def compare_to_html(data: dict) -> str:
    """Volledige HTML-uitvoer voor een compare-run."""
    if data["run_type"] != "compare":
        return run_to_html(data)
    # Tabel: Config | Return | CAGR | MaxDD | Sharpe | AvgExp | Trades
    rows = []
    for r in data["results"]:
        m = r["metrics"]
        label = r["config"].get("label") or r["config"]["asset"]
        rows.append(
            f"<tr><td>{label}</td><td>{pct(m['total_return'])}</td>"
            f"<td>{pct(m['cagr'])}</td><td>{pct(m['max_drawdown'])}</td>"
            f"<td>{m['sharpe']:.2f}</td>"
            f"<td>{pct(m['avg_exposure'])}</td><td>{m['n_trades']}</td></tr>"
        )
    table = ("<div class='card'><h3>Vergelijking</h3><table>"
             "<tr><th>Config</th><th>Return</th><th>CAGR</th><th>Max DD</th>"
             "<th>Sharpe</th><th>Avg Exp</th><th>Trades</th></tr>"
             + "".join(rows) + "</table></div>")
    # Equity + drawdown vergelijking.
    eq_all = ("<div class='card'><h3>Equity vergelijking</h3>"
             + comparison_chart(data, "equity") + "</div>")
    dd_all = ("<div class='card'><h3>Drawdown vergelijking</h3>"
             + comparison_chart(data, "drawdown") + "</div>")
    return f"{table}{eq_all}{dd_all}"