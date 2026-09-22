"""Backtest-runner voor de webui: een losse functie met nette output."""
from __future__ import annotations

import json

from hermes_bot.backtest import (
    Backtester,
    VolTargetStrategy,
    generate_synthetic_prices,
    load_prices,
)


def run_backtest(
    symbol: str = "SPY",
    period: str = "2y",
    use_live: bool = True,
    vol_target: float = 0.125,
    max_weight: float = 0.95,
) -> dict:
    """Draai een backtest en geef een JSON-serializable dict terug.

    use_live=True -> echte marktdata via yfinance; anders synthetic.
    """
    try:
        if use_live:
            prices = load_prices(symbol, period=period)
        else:
            prices = generate_synthetic_prices(n=500)
    except Exception as e:
        # Fallback naar synthetic bij netwerk-fout.
        prices = generate_synthetic_prices(n=500)
        symbol = f"{symbol} (synthetic — live-fout: {e})"

    strategy = VolTargetStrategy({"entity": symbol, "vol_target": vol_target})
    risk_cfg = {
        "risk": {
            "max_asset_weight": max_weight,
            "max_var_95": -0.20,
            "max_crash_prob": 0.30,
        }
    }
    r = Backtester().run(prices, strategy, risk_config=risk_cfg)

    # Pack naar JSON voor de webui-chart.
    n = len(r.equity_curve)
    # Sample tot max 400 punten voor de chart.
    step = max(1, n // 400)
    tx = [str(t.date()) for t in r.timestamps[::step]]
    equity = [round(e, 2) for e in r.equity_curve[::step]]

    # Benchmark-equity (buy-and-hold).
    close_first = float(prices["close"].iloc[0])
    bench = [round(r.initial_capital / close_first * c, 2) for c in prices["close"].iloc[::step]]

    return {
        "symbol": symbol,
        "period": period,
        "total_return": round(r.total_return, 4),
        "benchmark_return": round(r.benchmark_return, 4),
        "max_drawdown": round(r.max_drawdown, 4),
        "sharpe": round(r.sharpe_ratio, 3),
        "sortino": round(r.sortino_ratio, 3),
        "n_trades": r.n_trades,
        "win_rate": round(r.win_rate, 3),
        "initial_capital": r.initial_capital,
        "final_capital": round(r.final_capital, 2),
        "equity": equity,
        "benchmark": bench,
        "timestamps": tx,
    }


if __name__ == "__main__":
    print(json.dumps(run_backtest(), indent=2))