"""The integrated architecture pipeline.

This is the actual integration of ALL layers in one orchestrator.
The so-called "extensions" (bottleneck, regime, regional scores, orderflow)
are NO longer separate modules — they are real fusion-inputs and risk-inputs
that sit in the middle of the chain:

  24/7 server → webscraping → signals
       → [bottleneck agent + regime + regional scores + orderflow]  ← integrated
       → fusion model
       → risk engine (v2.1) + Monte Carlo
       → portfolio engine → execution

Each step gets a structured dict with the fields the next layer
expects, so the chain runs end-to-end and every intermediate value
is visible/loggable.
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
from hermes_bot.impact import ImpactAgent, LLMImpactAgent
from hermes_bot.portfolio import PortfolioState
from hermes_bot.risk_v2.montecarlo import MonteCarloEngineV2
from hermes_bot.risk_v2_1 import RiskEngineV21
from hermes_bot.rl.context import AssetContext, build_asset_context


@dataclass
class PipelineResult:
    """Full output of one chain cycle, with all intermediate values."""

    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    web_inputs: list[dict] = field(default_factory=list)
    fusion: dict = field(default_factory=dict)
    bottleneck_ratings: dict = field(default_factory=dict)
    regime: str = "unknown"
    regional_scores: dict = field(default_factory=dict)
    impacted: list[dict] = field(default_factory=list)
    risk_budget: float = 0.0
    effective_exposure: float = 0.0
    decision: dict = field(default_factory=dict)
    decisions: list[dict] = field(default_factory=list)  # per-entiteit RL/risk-output
    exposures: list[float] = field(default_factory=list)
    asset_contexts: list[dict] = field(default_factory=list)  # per-entiteit RL-input


def collect_web_inputs(
    speech: list[dict] | None = None,
    reports: list[dict] | None = None,
    alerts: list[dict] | None = None,
) -> list[dict]:
    """Fetch web inputs (offline demo if nothing is provided)."""
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
    """Convert raw web report dicts into ReportSignal objects.

    This is the correct integration: the extensions (RegionalScorer,
    BottleneckAnalyzer) work on ReportSignal objects, not on raw dicts.
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
    """Bottleneck agent as a fusion input (integrated, not standalone).

    Pulls question signals from web reports, links companies to
    supply-chain nodes, and gives per-company ratings as fusion inputs.
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
    """Regional scores as a fusion input (integrated)."""
    from hermes_bot.data.scraper import demo_report_records

    scorer = RegionalScorer()
    reports = _to_report_signals(reports or demo_report_records())
    scores = scorer.score_from_reports(reports)
    avg = sum(scores.values()) / len(scores) if scores else 0.0
    return {
        "source": "regional",
        "entity_id": "market",
        "sentiment": round(avg * 2.0 - 1.0, 4),  # 0..1 scores -> -1..1 direction
        "confidence": round(0.4 + 0.3 * (2 * avg - 1), 4) if avg else 0.3,
        "regional_scores": scores,
    }


def build_regime_input(features: dict | None = None) -> dict:
    """Regime detection as a risk input (integrated with v2.1)."""
    detector = RegimeDetector()
    features = features or {
        "vix": 18, "credit_spread": 1.2, "yield_curve": 0.3, "momentum": 0.04
    }
    regime = detector.detect(features)
    # Translate regime into a fusion sentiment + risk gate.
    regime_sent = {"bull": 0.5, "bear": -0.4, "highvol": -0.2, "crash": -0.8}.get(regime, 0.0)
    return {
        "source": "agent",  # regime-feature as agent-like macro-input
        "entity_id": "market",
        "sentiment": round(regime_sent, 4),
        "confidence": 0.6,
        "regime": regime,
    }


class Pipeline:
    """Orchestrates the full architecture chain (integrated)."""

    def __init__(self, config: dict | None = None) -> None:
        self.cfg = config or {}
        self.fusion = WeightedFusion()
        self.risk = RiskEngineV21(config or {})
        self.mc = MonteCarloEngineV2(seed=self.cfg.get("seed", 42), n_paths=1000)
        self.bottleneck_analyzer = BottleneckAnalyzer()
        # Impact agent: keyword matching, or with LLM interpretation as config that asks for it.
        if self.cfg.get("impact_use_llm"):
            self.impact_agent = LLMImpactAgent(
                model=self.cfg.get("impact_llm_model", "meta-llama/llama-3.1-8b-instruct:free"))
        else:
            self.impact_agent = ImpactAgent()
        self.portfolio = PortfolioState(cash=100_000.0)

    def run(
        self,
        price: float = 100.0,
        entity_id: str = "market",
        web_inputs: list[dict] | None = None,
        reports: list[dict] | None = None,
        regime_features: dict | None = None,
    ) -> PipelineResult:
        """Run one full cycle and return all intermediate values."""
        # 1. Web-inputs (speech/reports/alerts).
        inputs = web_inputs or collect_web_inputs()
        self.web_inputs = inputs

        # 2. Extensions as fusion INPUTS (integrated, not standalone).
        bottleneck_inputs = build_bottleneck_inputs(reports)
        regional_input = build_regional_scores_input(reports)
        regime_input = build_regime_input(regime_features)

        # 2b. IMPACT-AGENT: connect events to affected instruments.
        #     Determines WHICH stocks/bonds/ETFs get affected by the
        #     speech/report/alert, and provides per-entity fusion inputs.
        impact_inputs: list[dict] = []
        impacted: list[dict] = []
        impact_vectors: dict[str, dict] = {}  # entity -> impact-vector (categorie 1)
        for inp in inputs:
            text = inp.get("body") or inp.get("headline") or inp.get("summary", "")
            entity_id = inp.get("entity_id", "")
            # Skip only if there is NO text AND NO entity_id (nothing to match).
            if not text and not entity_id:
                continue
            result = self.impact_agent.analyze(
                text, source=inp.get("source", "news"),
                entity_id=entity_id,
            )
            if result.impacted:
                impacted.extend(result.impacted)
                impact_inputs.extend(
                    self.impact_agent.to_fusion_inputs(result, sentiment=inp.get("sentiment", 0.0))
                )
                # Impact vector (category 1) per affected entity.
                vec = {
                    "direction": result.direction,
                    "magnitude": result.magnitude,
                    "confidence": result.confidence,
                    "probability": result.probability,
                    "duration": result.duration,
                    "directness": result.directness,
                    "novelty": result.novelty,
                    "surprise": result.surprise,
                }
                for item in result.impacted:
                    impact_vectors[item["entity"]] = vec

        # Regional scores + regime/orderflow are RISK inputs, not fusion inputs.
        # They go to the risk engine via alpha_signals (below), not to fusion.
        all_inputs = inputs + bottleneck_inputs + impact_inputs

        # 3. Fusion combines all sources.
        fused = self.fusion.fuse(all_inputs)

        # 3b. Build an AssetContext per affected entity (RL-input)
        #     Impact-vector (categorie 1) + fusion (categorie 4) + bottleneck
        #     (category 5) + source info (category 2). This is what the RL model
        #     per asset ziet.
        asset_contexts: list[AssetContext] = []
        for item in impacted:
            entity = item.get("entity", "")
            if not entity:
                continue
            # Fusion per entity (only the impact input for that entity).
            ent_inputs = [i for i in impact_inputs if i.get("entity_id") == entity]
            ent_fused = self.fusion.fuse(ent_inputs) if ent_inputs else fused
            # Bottleneck for this entity (if present).
            ent_bn = next((i for i in bottleneck_inputs if i.get("entity_id") == entity), None)
            # Source info: from the original web input that this entity relates to.
            src = next((i for i in inputs if i.get("entity_id") == entity), None)
            ctx = build_asset_context(
                impact=impact_vectors.get(entity, item),
                fusion=ent_fused.model_dump() if hasattr(ent_fused, "model_dump") else ent_fused,
                bottleneck=ent_bn,
                source=src,
                entity=entity,
                asset_class=item.get("asset_class", ""),
            )
            asset_contexts.append(ctx)

        # 4. Risk engine (v2.1) with the v2.1 config.
        #    Each affected entity gets its OWN decision + risk approval, so
        #    multiple impact-agent outputs run separately through the RL/risk
        #    chain (e.g. EU AI plan -> MLST, ASML, STM each evaluated alone).
        equity = self.portfolio.cash + 0.0  # cash-only start
        mc = None
        if hasattr(self.risk, "hist_returns") and len(self.risk.hist_returns) >= 30:
            import numpy as np
            mc = self.mc.simulate(np.asarray(self.risk.hist_returns[-60:]), horizon=20)

        from hermes_bot.schemas import Action, ExitReason, RLRawDecision
        # Risk inputs: regional scores + regime/orderflow (not fusion inputs).
        regional_dict = regional_input.get("regional_scores", {})
        regional_avg = 0.0
        if regional_dict:
            regional_avg = sum(regional_dict.values()) / len(regional_dict)
        alpha_signals = {
            "regional_score": round(regional_avg, 4),
            "regime": regime_input.get("regime", "unknown"),
            "regime_sentiment": regime_input.get("sentiment", 0.0),
        }

        decisions: list[dict] = []
        exposures: list[float] = []
        # If the impact agent found entities, decide per entity; else one
        # market-level decision.
        entities = [c.entity for c in asset_contexts] or [entity_id]
        for ent in entities:
            ent_ctx = next((c for c in asset_contexts if c.entity == ent), None)
            ent_sent = ((ent_ctx.market_sentiment - 0.5) * 2.0 if ent_ctx
                        else fused.emotie.get("sentiment", 0.0))
            ent_conf = ent_ctx.sentiment_confidence if ent_ctx else fused.zekerheid
            decision = RLRawDecision(
                entity=ent, action=Action.BUY,
                intent_to_alloc=1.0 if ent_sent >= 0 else 0.0,
                zekerheid=ent_conf, rationale="pipeline-integrated",
                timestamp=datetime.now(UTC), exit_reason=ExitReason.NONE,
            )
            exposure, breakdown = self.risk.approve(
                decision, self.portfolio, equity, mc, alpha_signals=alpha_signals)
            decisions.append(decision.model_dump())
            exposures.append(round(exposure, 4))
        # Primary decision = the first (or market-level) one for the result.
        decision = RLRawDecision(**decisions[0]) if decisions else None
        exposure = exposures[0] if exposures else 0.0

        # 5. Return bottleneck ratings (for logging/attribution).
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
            impacted=impacted,
            risk_budget=round(breakdown.risk_budget, 4),
            effective_exposure=round(exposure, 4),
            decision=decision.model_dump() if decision else {},
            decisions=decisions,
            exposures=exposures,
            asset_contexts=[c.to_dict() for c in asset_contexts],
        )

    def record_equity(self, equity: float) -> None:
        self.risk.update_drawdown(equity)


def run_pipeline(price: float = 100.0) -> dict:
    """Short entry point for CLI/tests."""
    return Pipeline().run(price=price).__dict__


if __name__ == "__main__":
    import json
    print(json.dumps(run_pipeline(), indent=2, default=str))
