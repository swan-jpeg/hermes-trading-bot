"""De geïntegreerde architectuur-pipeline.

Dit is de daadwerkelijke integratie van ALLE lagen in één orchestrator.
De zogenaamde "uitbreidingen" (bottleneck, regime, regionale scores, orderflow)
zijn hier GEEN losse modules meer — ze zijn echte fusion-inputs en risk-inputs
die midden in de keten zitten:

  24/7 server → webscraping → signalen
       → [bottleneck agent + regime + regionale scores + orderflow]  ← geïntegreerd
       → fusion model
       → risk engine (v2.1) + Monte Carlo
       → portfolio engine → executie

Elke stap krijgt een gestructureerd dict met de velden die de volgende laag
verwacht, zodat de keten end-to-end draait en elke tussenliggende waarde
zichtbaar/logbaar is.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from hermes_bot.expansions import (
    BottleneckAgent,
    BottleneckAnalyzer,
    RegimeDetector,
    RegionalScorer,
)
from hermes_bot.fusion import WeightedFusion
from hermes_bot.portfolio import PortfolioState
from hermes_bot.risk_v2.montecarlo import MonteCarloEngineV2
from hermes_bot.risk_v2_1 import RiskEngineV21


@dataclass
class PipelineResult:
    """Volledige uitvoer van één keten-cyclus, met alle tussenwaardes."""

    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    web_inputs: list[dict] = field(default_factory=list)
    fusion: dict = field(default_factory=dict)
    bottleneck_ratings: dict = field(default_factory=dict)
    regime: str = "unknown"
    regional_scores: dict = field(default_factory=dict)
    risk_budget: float = 0.0
    effective_exposure: float = 0.0
    decision: dict = field(default_factory=dict)


def collect_web_inputs(
    speech: list[dict] | None = None,
    reports: list[dict] | None = None,
    alerts: list[dict] | None = None,
) -> list[dict]:
    """Haal web-inputs op (offline-demo als niets aangeleverd wordt)."""
    from hermes_bot.data.scraper import (
        demo_alert_records,
        demo_report_records,
        demo_speech_records,
        web_records_to_inputs,
    )

    if speech is None and reports is None and alerts is None:
        speech, reports, alerts = (
            demo_speech_records(), demo_report_records(), demo_alert_records()
        )
    return web_records_to_inputs(speech, reports, alerts, entity_id="market")


def _to_report_signals(reports: list[dict] | None) -> list:
    """Zet ruwe webrapport-dicts om naar ReportSignal-objecten.

    Dit is de correcte integratie: de uitbreidingen (RegionalScorer,
    BottleneckAnalyzer) werken op ReportSignal-objecten, niet op ruwe dicts.
    """
    from hermes_bot.schemas import SourceKind
    from hermes_bot.signals.textual import ReportSignal

    out = []
    for r in reports or []:
        out.append(ReportSignal(
            source=SourceKind.REPORT,
            source_name=r.get("source", "report"),
            entity_id="market",
            timestamp=datetime.now(UTC),
            confidence=r.get("confidence", 0.5),
            report_type=r.get("report_type", "business"),
            summary=r.get("body", r.get("headline", "")),
            body=r.get("body", ""),
            headline=r.get("headline", ""),
            extracted_topics=[],
        ))
    return out


def build_bottleneck_inputs(reports: list[dict] | None = None) -> list[dict]:
    """Bottleneck-agent als fusion-input (geïntegreerd, niet los).

    Haalt vraagsignalen uit webrapporten, koppelt bedrijven aan
    supply-chain knooppunten, en geeft per-bedrijf ratings als fusie-inputs.
    """
    from hermes_bot.data.scraper import demo_report_records

    agent = BottleneckAgent()
    reports = reports or demo_report_records()
    entity_map = {
        "NVDA": ["semiconductor"],
        "AIRTECH": ["actuators"],
        "POWCO": ["power"],
        "APLE": ["semiconductor"],
        "TSLA": ["actuators", "battery"],
    }
    return agent.run_pipeline(reports, entity_map, market_sentiment=0.0)


def build_regional_scores_input(reports: list[dict] | None = None) -> dict:
    """Regionale scores als fusion-input (geïntegreerd)."""
    from hermes_bot.data.scraper import demo_report_records

    scorer = RegionalScorer()
    reports = _to_report_signals(reports or demo_report_records())
    scores = scorer.score_from_reports(reports)
    avg = sum(scores.values()) / len(scores) if scores else 0.0
    return {
        "source": "regional",
        "entity_id": "market",
        "sentiment": round(avg * 2.0 - 1.0, 4),  # 0..1 scores -> -1..1 richting
        "confidence": round(0.4 + 0.3 * (2 * avg - 1), 4) if avg else 0.3,
        "regional_scores": scores,
    }


def build_regime_input(features: dict | None = None) -> dict:
    """Regime-detectie als risk-input (geïntegreerd met v2.1)."""
    detector = RegimeDetector()
    features = features or {
        "vix": 18, "credit_spread": 1.2, "yield_curve": 0.3, "momentum": 0.04
    }
    regime = detector.detect(features)
    # Vertaal regime naar een fusie-sentiment + risk-gate.
    regime_sent = {"bull": 0.5, "bear": -0.4, "highvol": -0.2, "crash": -0.8}.get(regime, 0.0)
    return {
        "source": "agent",  # regime-feature als agent-achtige macro-input
        "entity_id": "market",
        "sentiment": round(regime_sent, 4),
        "confidence": 0.6,
        "regime": regime,
    }


class Pipeline:
    """Orchestreert de volledige architectuurketen (geïntegreerd)."""

    def __init__(self, config: dict | None = None) -> None:
        self.cfg = config or {}
        self.fusion = WeightedFusion()
        self.risk = RiskEngineV21(config or {})
        self.mc = MonteCarloEngineV2(seed=self.cfg.get("seed", 42), n_paths=1000)
        self.bottleneck_analyzer = BottleneckAnalyzer()
        self.portfolio = PortfolioState(cash=100_000.0)

    def run(
        self,
        price: float = 100.0,
        entity_id: str = "market",
        web_inputs: list[dict] | None = None,
        reports: list[dict] | None = None,
        regime_features: dict | None = None,
    ) -> PipelineResult:
        """Draai één volledige cyclus en geef alle tussenwaardes."""
        # 1. Web-inputs (speech/reports/alerts).
        inputs = web_inputs or collect_web_inputs()
        self.web_inputs = inputs

        # 2. Outbreidingen als fusion-INPUTS (geïntegreerd, niet los).
        bottleneck_inputs = build_bottleneck_inputs(reports)
        regional_input = build_regional_scores_input(reports)
        regime_input = build_regime_input(regime_features)

        all_inputs = inputs + bottleneck_inputs + [regional_input, regime_input]

        # 3. Fusion combineert alle bronnen.
        fused = self.fusion.fuse(all_inputs)

        # 4. Risk engine (v2.1) met de v2.1-config.
        decision_sent = fused.emotie.get("sentiment", 0.0)
        equity = self.portfolio.cash + 0.0  # cash-only start
        mc = None
        if hasattr(self.risk, "hist_returns") and len(self.risk.hist_returns) >= 30:
            import numpy as np
            mc = self.mc.simulate(np.asarray(self.risk.hist_returns[-60:]), horizon=20)

        from hermes_bot.schemas import Action, ExitReason, RLRawDecision
        decision = RLRawDecision(
            entity=entity_id, action=Action.BUY,
            intent_to_alloc=1.0 if decision_sent >= 0 else 0.0,
            zekerheid=fused.zekerheid, rationale="pipeline-integrated",
            timestamp=datetime.now(UTC), exit_reason=ExitReason.NONE,
        )
        exposure, breakdown = self.risk.approve(decision, self.portfolio, equity, mc)

        # 5. Bottleneck-ratings teruggeven (voor logging/attributie).
        ratings = {
            i["entity_id"]: round(i["bottleneck_score"], 4)
            for i in bottleneck_inputs if "bottleneck_score" in i
        }

        return PipelineResult(
            web_inputs=all_inputs,
            fusion=fused.model_dump(),
            bottleneck_ratings=ratings,
            regime=regime_input.get("regime", "unknown"),
            regional_scores=regional_input.get("regional_scores", {}),
            risk_budget=round(breakdown.risk_budget, 4),
            effective_exposure=round(exposure, 4),
            decision=decision.model_dump(),
        )

    def record_equity(self, equity: float) -> None:
        self.risk.update_drawdown(equity)


def run_pipeline(price: float = 100.0) -> dict:
    """Kort entry-point voor CLI/tests."""
    return Pipeline().run(price=price).__dict__


if __name__ == "__main__":
    import json
    print(json.dumps(run_pipeline(), indent=2, default=str))