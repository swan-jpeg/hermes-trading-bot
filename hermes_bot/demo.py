"""Demo pipeline: walks through the whole architecture chain.

data -> fusie -> RL (prijs-bewust) -> risico (incl. Monte Carlo) -> executie (paper).

Toont de architectuurverbeteringen voor winst nemen:
- Positie met entry-prijs wordt doorgegeven aan de RL-laag.
- Take-profit / stop-loss / trailing-stop exit-logica.
- Monte Carlo-resultaat begrenst de allocatie in de risico-engine.
Gebruik: `uv run python -m hermes_bot.demo`
"""
from __future__ import annotations

from datetime import UTC, datetime

from hermes_bot.execution import ExecutionEngine, Order, PaperBroker
from hermes_bot.expansions import BottleneckAnalyzer, RegimeDetector
from hermes_bot.fusion import WeightedFusion
from hermes_bot.portfolio import PortfolioState
from hermes_bot.risk import RiskEngine
from hermes_bot.rl import RLFusionModel
from hermes_bot.schemas import Position
from hermes_bot.simulation import MonteCarloEngine


def run_demo() -> dict:
    """Simulate a decision cycle with an existing position + Monte Carlo."""
    # 0. Een bestaande open positie (entry 100, nu 118 -> +18%, boven take-profit 15%).
    pos = Position(
        entity="AAPL",
        qty=100.0,
        entry_price=100.0,
        entry_time=datetime.now(UTC),
        take_profit_pct=0.15,
        stop_loss_pct=0.08,
        trailing_stop_pct=0.05,
    )
    portfolio = PortfolioState(cash=100_000.0, positions={"AAPL": pos}, regime="bull")
    current_price = 118.0

    # 1. Fusion inputs (would come from the data layer).
    inputs = [
        {"source": "agent", "entity_id": "AAPL", "sentiment": 0.4, "confidence": 0.8},
        {"source": "market", "entity_id": "AAPL", "sentiment": 0.2, "confidence": 0.7},
        {"source": "regional", "entity_id": "AAPL", "sentiment": 0.1, "confidence": 0.5},
    ]
    fusion = WeightedFusion().fuse(inputs)

    # 2. Monte Carlo (historische rendementen simuleren).
    import numpy as np

    rng = np.random.default_rng(7)
    hist = rng.normal(0.0005, 0.02, 250)  # ~250 dagen historie
    mc = MonteCarloEngine(n_paths=5000).simulate(hist, horizon=30)

    # 3. RL decision (price-aware: sees the position + profit -> take-profit).
    rl = RLFusionModel()
    decision = rl.decide(fusion.model_dump(), {"entity_id": "AAPL"}, portfolio, current_price)

    # 4. Risico-engine keurt goed (incl. Monte Carlo-begrenzing).
    risk = RiskEngine({"risk": {"max_asset_weight": 0.10}})
    approval = risk.approve(decision, portfolio, mc)

    # 5. Execution (paper) if approved.
    fills: list[dict] = []
    if approval.approved:
        eng = ExecutionEngine(PaperBroker())
        target_qty = approval.target_alloc.get("AAPL", 0.0)
        # Sell the full position on exit; otherwise buy/reduce based on alloc.
        if decision.action.value == "sell" and decision.exit_reason.value != "none":
            side, qty = "sell", int(pos.qty)
        else:
            side, qty = ("buy" if target_qty >= 0 else "sell"), int(abs(target_qty) * 1000)
        if qty > 0:
            eng.execute([Order(asset="AAPL", side=side, qty=qty)])
            fills = eng.fills()

    # 6. Uitbreidingen: regime + bottleneck.
    regime = RegimeDetector().detect(
        {"vix": 18, "credit_spread": 1.2, "yield_curve": 0.3, "momentum": 0.05}
    )
    ba = BottleneckAnalyzer()
    ba.add_node("actuator-leverancier", demand=0.9, capacity=0.4, players=3)
    ba.add_edge("actuator-leverancier", "robotica")
    ba.add_edge("actuator-leverancier", "EV")
    bottlenecks = ba.top_bottlenecks(3)

    return {
        "position": {"entity": pos.entity, "entry": pos.entry_price, "current": current_price,
                     "pnl_pct": round(pos.unrealized_pnl_pct(current_price), 4)},
        "fusion": fusion.model_dump(),
        "monte_carlo": {"var_95": round(mc.var_95, 4), "es_95": round(mc.expected_shortfall_95, 4),
                        "crash_prob": round(mc.crash_probability, 4)},
        "rl_decision": decision.model_dump(),
        "risk_approval": approval.model_dump(),
        "fills": fills,
        "regime": regime,
        "bottlenecks": bottlenecks,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run_demo(), indent=2, default=str))