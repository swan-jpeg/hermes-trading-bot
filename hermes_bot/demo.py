"""Demo-pipeline: doorloopt de hele architectuurketen.

data -> fusie -> RL -> risico -> executie (paper).
Gebruik: `uv run python -m hermes_bot.demo`
"""
from __future__ import annotations

from hermes_bot.execution import ExecutionEngine, Order, PaperBroker
from hermes_bot.expansions import BottleneckAnalyzer, RegimeDetector
from hermes_bot.fusion import WeightedFusion
from hermes_bot.portfolio import PortfolioState
from hermes_bot.risk import RiskEngine
from hermes_bot.rl import RLFusionModel


def run_demo() -> dict:
    """Simuleer één beslissingscyclus met voorbeelddata."""
    # 1. Fusie-inputs (zouden uit data-laag komen).
    inputs = [
        {"source": "agent", "entity_id": "AAPL", "sentiment": 0.6, "confidence": 0.8},
        {"source": "market", "entity_id": "AAPL", "sentiment": 0.3, "confidence": 0.7},
        {"source": "regional", "entity_id": "AAPL", "sentiment": 0.1, "confidence": 0.5},
    ]
    fusion = WeightedFusion().fuse(inputs)

    # 2. RL-besluit (voorstel).
    rl = RLFusionModel()
    decision = rl.decide(fusion.model_dump(), {"entity_id": "AAPL"}, {"regime": "bull"})

    # 3. Risico-engine keurt goed (of niet).
    risk = RiskEngine({"risk": {"max_asset_weight": 0.10}})
    approval = risk.approve(decision, PortfolioState(regime="bull"))

    # 4. Executie (paper) als goedgekeurd.
    fills: list[dict] = []
    if approval.approved:
        eng = ExecutionEngine(PaperBroker())
        qty = int(approval.target_alloc.get("AAPL", 0.0) * 1000)
        if qty > 0:
            eng.execute([Order(asset="AAPL", side="buy", qty=qty)])
            fills = eng.fills()

    # 5. Uitbreidingen: regime + bottleneck.
    regime = RegimeDetector().detect(
        {"vix": 18, "credit_spread": 1.2, "yield_curve": 0.3, "momentum": 0.05}
    )
    ba = BottleneckAnalyzer()
    ba.add_node("actuator-leverancier", demand=0.9, capacity=0.4, players=3)
    ba.add_edge("actuator-leverancier", "robotica")
    ba.add_edge("actuator-leverancier", "EV")
    bottlenecks = ba.top_bottlenecks(3)

    return {
        "fusion": fusion.model_dump(),
        "rl_decision": decision.model_dump(),
        "risk_approval": approval.model_dump(),
        "fills": fills,
        "regime": regime,
        "bottlenecks": bottlenecks,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run_demo(), indent=2, default=str))
