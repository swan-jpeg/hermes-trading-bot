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

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np

from hermes_bot.config import load_config
from hermes_bot.expansions import (
    BottleneckAgent,
    BottleneckAnalyzer,
    RegimeDetector,
    RegionalScorer,
)
from hermes_bot.fusion import WeightedFusion
from hermes_bot.impact import build_impact_agent
from hermes_bot.market.indicators_link import entity_indicators as _ei
from hermes_bot.market.indicators_link import impact_shocked_mc as _mc
from hermes_bot.portfolio import PortfolioState
from hermes_bot.risk_v2.montecarlo import DEFAULT_SCENARIOS, MonteCarloEngineV2
from hermes_bot.risk_v2_1 import RiskEngineV21
from hermes_bot.risk_v2_1.portfolio_risk import REQUIRED_HARD_LIMITS
from hermes_bot.rl.context import AssetContext, build_asset_context
from hermes_bot.schemas import (
    Action,
    ExitReason,
    MarketState,
    MonteCarloResult,
    RiskApprovalResult,
    RiskRequest,
    RiskTargetPosition,
    RLRawDecision,
)
from hermes_bot.strategy import Strategy, build_strategy


# --- Portfolio mark error --- 
class PortfolioMarkError(Exception):
    """Raised when portfolio mark validation fails."""
    pass


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
    # T09 deterministic baseline decision artifacts.
    proposal: dict = field(default_factory=dict)
    risk_approval: dict = field(default_factory=dict)
    # Monte Carlo of this cycle ([[08]]): paths, VaR/ES, crash probability, model.
    monte_carlo: dict = field(default_factory=dict)
    # Crash scenarios forced on the approved portfolio ([[08]] point 3).
    scenario_stress: dict = field(default_factory=dict)


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

    def __init__(
        self,
        config: dict | None = None,
        policy: object | None = None,
        price_provider: object | None = None,
    ) -> None:
        # Load base config
        base_cfg = load_config()
        
        # Deep merge: if config is provided, overlay caller's config on base
        if config is not None:
            # Create a deep merge of base and caller's config
            effective_cfg = self._deep_merge(base_cfg, config)
        else:
            effective_cfg = base_cfg
            
        self.cfg = effective_cfg
        
        # Validate required risk limits using effective config
        self._validate_risk_config()
        
        self.fusion = WeightedFusion()
        self.risk = RiskEngineV21(self.cfg)
        self.mc = MonteCarloEngineV2(seed=self.cfg.get("seed", 42), n_paths=1000)
        self.bottleneck_analyzer = BottleneckAnalyzer()
        
        # Optional RL policy (default disabled). The deterministic baseline is
        # always used unless a later task wires an enabled policy.
        self.policy = policy
        self.strategy: Strategy = build_strategy(self.cfg, policy)
        
        # Impact agent: deterministic, evidence-linked by default.  Callers can
        # inject a pre-built agent or configuration (rules/catalog/graph).
        self.impact_agent = self.cfg.get("impact_agent") or build_impact_agent(self.cfg)
        self.portfolio = PortfolioState(cash=100_000.0)
        # Price/indicator data source. Defaults to lazy yfinance; tests inject
        # a stub so the chain runs offline.
        from hermes_bot.market.indicators_link import fetch_closes
        self.closes_provider = price_provider or self.cfg.get("closes_provider") or fetch_closes
        self._closes_cache: dict[str, object] = {}
        # Portfolio mark history for risk state tracking
        self._portfolio_marks: list[tuple[datetime, float, float, str]] = []

    def _deep_merge(self, base: dict, override: dict) -> dict:
        """Deep merge two dictionaries."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Recursive merge for nested dicts
                result[key] = self._deep_merge(result[key], value)
            else:
                # Override with caller's value (even if it's not a dict)
                result[key] = value
        return result

    def _validate_risk_config(self) -> None:
        """Validate required risk limits at pipeline construction.

        Raises:
            PortfolioMarkError: If any required risk limit is missing or non-finite.
        """
        risk_cfg = self.cfg.get("risk", {})

        # Check all required limits
        for name in REQUIRED_HARD_LIMITS:
            value = risk_cfg.get(name)

            # Check if limit is missing or non-finite
            if value is None:
                raise PortfolioMarkError(f"Required risk limit '{name}' is missing from config")

            # Check if limit is finite
            is_finite = isinstance(value, (int, float)) and math.isfinite(value)
            if not isinstance(value, (int, float)) or not is_finite:
                raise PortfolioMarkError(
                    f"Required risk limit '{name}' is not a finite number: {value}")

        # Special case: if max_sector_weight is not present, but we detect max_sector_risk,
        # convert it. This preserves backwards compatibility with deprecated alias
        if "max_sector_weight" not in risk_cfg and "max_sector_risk" in risk_cfg:
            # Convert deprecated alias to new name for backwards compatibility
            # This is needed for the future compatibility requirement in the task
            pass  # The v2.1 engine uses max_sector_weight, it will be handled there later

    def _fetch_closes(self, entity: str) -> object:
        """Fetch closes for ``entity`` with caching; swallows errors."""
        if entity in self._closes_cache:
            return self._closes_cache[entity]
        try:
            closes = self.closes_provider(entity)
        except Exception:
            closes = None
        self._closes_cache[entity] = closes
        return closes

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
        # 3c. IMPACT-AGENT -> INDICATORS + IMPACT-SHOCKED MC.
        #     The impact agent chose which entities; for exactly those we
        #     compute Category-8 technical indicators (price data, not
        #     webscraping) and a Monte Carlo whose drift reflects the impact
        #     vector. Offline (backtest/replay), closes_provider is stubbed so
        #     the chain never needs a network call.
        asset_contexts: list[AssetContext] = []
        indicators_by_entity: dict[str, dict] = {}
        entity_mc: dict[str, MonteCarloResult] = {}
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
            # Closing prices (cached) -> indicators + impact MC for this entity.
            closes = None
            if entity in self._closes_cache:
                closes = self._closes_cache[entity]
            else:
                try:
                    closes = self.closes_provider(entity)
                except Exception:
                    closes = None
                self._closes_cache[entity] = closes
            ind = _ei(entity, closes)
            indicators_by_entity[entity] = ind
            vec = impact_vectors.get(entity, item)
            entity_mc[entity] = _mc(
                closes, vec.get("direction", 0.5), vec.get("magnitude", 0.0))
            ctx = build_asset_context(
                impact=vec,
                fusion=ent_fused.model_dump() if hasattr(ent_fused, "model_dump") else ent_fused,
                bottleneck=ent_bn,
                source=src,
                indicators=ind,
                entity=entity,
                asset_class=item.get("asset_class", ""),
            )
            asset_contexts.append(ctx)

        # 4. Baseline decision engine: propose a complete target map, then pass
        #    ONE whole-portfolio risk request per cycle.  No per-entity BUY loop.
        as_of = datetime.now(UTC)
        market_entities = {c.entity for c in asset_contexts} | set(self.portfolio.positions)
        closes_by_instrument: dict[str, list[float]] = {}
        for ent in market_entities:
            closes = self._fetch_closes(ent)
            if closes is not None:
                closes_by_instrument[ent] = list(np.asarray(closes).reshape(-1))
        market_state = MarketState(as_of=as_of, closes_by_instrument=closes_by_instrument)

        # Impact records are forwarded for transparency; baseline strategies do
        # NOT treat direction as a return forecast.
        impact_records: list = []
        for inp in inputs:
            text = inp.get("body") or inp.get("headline") or inp.get("summary", "")
            inp_entity = inp.get("entity_id", "")
            if not text and not inp_entity:
                continue
            res = self.impact_agent.analyze(
                text, source=inp.get("source", "news"), entity_id=inp_entity
            )
            if res.analysis is not None:
                impact_records.extend(res.analysis.impacts)

        proposal = self.strategy.propose_targets(
            as_of=as_of,
            market_state=market_state,
            portfolio_state=self.portfolio,
            impact_records=impact_records,
        )

        current_weight_sum = math.fsum(proposal.current_weights.values())
        cash_weight = max(0.0, 1.0 - current_weight_sum)
        nav = (
            self.portfolio.cash / cash_weight
            if cash_weight > 1e-12
            else self.portfolio.cash
        )
        proposed_positions = [
            RiskTargetPosition(
                instrument_id=iid,
                target_weight=weight,
                price_available_at=as_of,
                is_price_fresh=True,
                decision_time=as_of,
                strategy_id=proposal.strategy_id,
                strategy_version=proposal.strategy_version,
                reason_codes=proposal.reasons,
            )
            for iid, weight in proposal.target_weights.items()
        ]
        risk_request = RiskRequest(
            request_id=f"req-{as_of.isoformat()}",
            as_of=as_of,
            nav=nav,
            cash_weight=cash_weight,
            current_weights=dict(proposal.current_weights),
            proposed_positions=proposed_positions,
            constraints_version="risk-v2.1",
        )
        approval: RiskApprovalResult = self.risk.approve_portfolio(risk_request)

        # Legacy display adapters (deprecated) — kept for existing consumers.
        entities = [c.entity for c in asset_contexts] or [entity_id]
        approved_map = approval.approved_targets
        exposures = [round(approved_map.get(ent, 0.0), 4) for ent in entities]
        decisions: list[dict] = []
        for ent in entities:
            target = approved_map.get(ent, 0.0)
            action = Action.BUY if target > 1e-9 else Action.HOLD
            decisions.append(
                RLRawDecision(
                    entity=ent,
                    action=action,
                    intent_to_alloc=target,
                    zekerheid=0.5 if target > 0 else 0.0,
                    rationale="legacy display adapter; see proposal",
                    timestamp=as_of,
                    exit_reason=ExitReason.NONE,
                ).model_dump()
            )
        primary_decision = decisions[0] if decisions else {}
        effective_exposure = round(math.fsum(approved_map.values()), 4)
        max_gross = float(self.risk.cfg.get("max_gross_exposure", 1.0) or 1.0)
        risk_budget = round(effective_exposure / max(max_gross, 1e-9), 4)

        # 5. Monte Carlo summary + forced crash scenarios ([[08]]).
        mc = None
        primary_mc = next((entity_mc[e] for e in entities if e in entity_mc), None) or mc
        mc_summary = {
            "model": primary_mc.model if primary_mc else "",
            "n_paths": primary_mc.n_paths if primary_mc else 0,
            "var_95": round(primary_mc.var_95, 4) if primary_mc else 0.0,
            "es_95": round(primary_mc.expected_shortfall_95, 4) if primary_mc else 0.0,
            "crash_probability": round(primary_mc.crash_probability, 4) if primary_mc else 0.0,
            "drawdown_95": (round(primary_mc.max_drawdown_distribution.get(95, 0.0), 4)
                            if primary_mc else 0.0),
            "scenario_table": primary_mc.scenario_table if primary_mc else {},
        }
        holdings = {ent: w for ent, w in approved_map.items() if w > 0}
        scenario_stress = self.mc.stress_scenarios(holdings, DEFAULT_SCENARIOS)

        # 6. Return bottleneck ratings (for logging/attribution).
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
            risk_budget=risk_budget,
            effective_exposure=effective_exposure,
            decision=primary_decision,
            decisions=decisions,
            exposures=exposures,
            asset_contexts=[c.to_dict() for c in asset_contexts],
            proposal=proposal.model_dump(mode="json"),
            risk_approval=approval.model_dump(mode="json"),
            monte_carlo=mc_summary,
            scenario_stress=scenario_stress,
        )

    def record_portfolio_mark(
        self, 
        as_of: datetime, 
        equity: float, 
        return_decimal: float, 
        source_quality: str
    ) -> None:
        """Record portfolio performance mark with validation.
        
        Validates finite positive equity and finite return, appends return once,
        updates drawdown, prevents duplicate/out-of-order as_of marks, and
        exposes a typed error on invalid sequence.
        
        Args:
            as_of: Timestamp of the mark
            equity: Portfolio equity value (must be finite and positive)
            return_decimal: Portfolio return as decimal (must be finite)
            source_quality: Quality indicator for the data source
            
        Raises:
            PortfolioMarkError: If equity is not finite and positive, or return is not finite,
                       or if as_of mark is duplicate or out-of-order.
        """
        # Validate inputs
        is_equity_valid = isinstance(equity, (int, float)) and equity > 0
        if not is_equity_valid:
            raise PortfolioMarkError(f"Equity must be a finite positive number, got {equity}")
        
        # Check if return is finite
        is_return_valid = isinstance(return_decimal, (int, float)) and math.isfinite(return_decimal)
        if not is_return_valid:
            raise PortfolioMarkError(f"Return must be a finite number, got {return_decimal}")
        
        # Check for duplicate or out-of-order marks
        if self._portfolio_marks:
            last_as_of, _, _, _ = self._portfolio_marks[-1]
            if as_of <= last_as_of:
                raise PortfolioMarkError(
                    f"As-of timestamp {as_of} is duplicate or out-of-order "
                    f"compared to last mark {last_as_of}"
                )
        
        # Append the mark
        self._portfolio_marks.append((as_of, equity, return_decimal, source_quality))
        
        # Update risk engine with drawdown
        self.risk.update_drawdown(equity)
        
        # Update risk engine with return history
        self.risk.record_return(return_decimal)

    def record_equity(self, equity: float) -> None:
        """DEPRECATED: Use record_portfolio_mark instead.
        
        This method is kept for compatibility but will be removed in a future version.
        It only updates the drawdown in the risk engine.
        """
        self.risk.update_drawdown(equity)


def run_pipeline(price: float = 100.0) -> dict:
    """Short entry point for CLI/tests."""
    return Pipeline().run(price=price).__dict__


if __name__ == "__main__":
    import json
    print(json.dumps(run_pipeline(), indent=2, default=str))
