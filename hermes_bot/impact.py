"""Impact Agent — links events to affected instruments.

The missing link in the chain: it decides WHICH instruments (stocks, bonds,
ETFs) are affected by an event (speech, quarterly report, government
spending, news). This is a PARALLEL step before the fusion model:

    webscraping (event)
        |
    IMPACT-AGENT  <- decides: "this hits NVDA, TSLA, AGG"
        |
    per affected entity: vector {entity, asset_class, sentiment, confidence}
        |
    RL model (per entity: which instrument + weight + confidence)
        |
    risk engine + Monte Carlo -> portfolio -> execution

Without this agent the RL model is "blind" — it does not know which stocks
are hit by a speech or report. This agent provides that link.

Two modes:
- ImpactAgent: keyword/sector matching (fast, free, offline, always).
- LLMImpactAgent: uses your own LLM with the impact-agent skill
  (impact_agent/skill/impact-agent-skill.md) to understand the context.
  The API key/base_url come from .env, never from code.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass, field

# Sector/topic -> affected instruments (tickers + asset class).
# Extensible: add keywords per sector.
_SECTOR_MAP: dict[str, dict] = {
    "semiconductor": {
        "keywords": ["chip", "semiconductor", "halfgeleider", "nvidia", "tsmc",
                     "ai chip", "gpu", "datacenter"],
        "tickers": ["NVDA", "AMD", "INTC", "TSM", "AVGO"],
        "asset_class": "stock",
    },
    "actuators": {
        "keywords": ["actuator", "robot", "robotics", "automation", "motion"],
        "tickers": ["TSLA", "AIRTECH", "FANUY"],
        "asset_class": "stock",
    },
    "power": {
        "keywords": ["power", "energy", "electricity", "grid", "utility", "energie"],
        "tickers": ["POWCO", "NEE", "DUK", "XLE"],
        "asset_class": "stock",
    },
    "battery": {
        "keywords": ["battery", "lithium", "ev", "electric vehicle", "accu"],
        "tickers": ["TSLA", "QS", "ALB", "LIT"],
        "asset_class": "stock",
    },
    "government": {
        "keywords": ["overheid", "government", "uitgave", "spending", "stimulus",
                     "begroting", "budget", "infrastructuur", "infrastructure"],
        "tickers": ["AGG", "TLT", "LQD", "XLI"],
        "asset_class": "bond",
    },
    "macro": {
        "keywords": ["inflation", "rente", "interest rate", "cpi", "werkloosheid",
                     "unemployment", "gdp", "economie", "economy"],
        "tickers": ["SPY", "QQQ", "IWM", "EFA", "AGG"],
        "asset_class": "etf",
    },
    "consumer": {
        "keywords": ["consumer", "retail", "winkel", "verkoop", "sales", "besteding"],
        "tickers": ["AMZN", "WMT", "COST", "XLY"],
        "asset_class": "stock",
    },
    "healthcare": {
        "keywords": ["health", "pharma", "ziekenhuis", "medisch", "drug", "vaccin"],
        "tickers": ["JNJ", "PFE", "UNH", "XLV"],
        "asset_class": "stock",
    },
    "finance": {
        "keywords": ["bank", "finance", "financieel", "krediet", "credit", "loan"],
        "tickers": ["JPM", "BAC", "GS", "XLF"],
        "asset_class": "stock",
    },
    "tech": {
        "keywords": ["tech", "software", "cloud", "ai", "kunstmatige intelligentie",
                     "internet", "platform"],
        "tickers": ["MSFT", "AAPL", "GOOGL", "META", "XLK"],
        "asset_class": "stock",
    },
}


@dataclass
class ImpactResult:
    """Output of the impact agent: which instruments are affected."""

    event: str
    matched_sectors: list[str] = field(default_factory=list)
    impacted: list[dict] = field(default_factory=list)  # [{entity, asset_class, reason}]
    confidence: float = 0.0
    # Impact vector (category 1 of the RL inputs): per analysis, normalized.
    direction: float = 0.5       # 0=negative, 1=positive
    magnitude: float = 0.0        # how large the expected impact is
    probability: float = 0.0     # chance the impact actually occurs
    duration: float = 0.5        # how long the impact lasts
    directness: float = 0.0      # 1=directly hit, 0=indirectly
    novelty: float = 0.0         # how new/unexpected
    surprise: float = 0.0        # deviation from what was already expected


class ImpactAgent:
    """Links an event to affected instruments via keyword matching."""

    def __init__(self, sector_map: dict | None = None) -> None:
        self.sector_map = sector_map or _SECTOR_MAP

    def analyze(self, event: str, source: str = "news",
                entity_id: str = "") -> ImpactResult:
        """Determine which instruments are hit by this event.

        event: the text of the event (speech, report, government spending).
        source: speech | report | alert | government.
        entity_id: optional entity (e.g. "trump") that also serves as a hint.
        """
        text = (event + " " + entity_id).lower()
        # Speech of a world leader (trump, biden, putin, ...) -> macro impact.
        if source == "speech" and any(
            leader in entity_id.lower()
            for leader in ("trump", "biden", "putin", "leader", "president", "minister")
        ):
            text += " inflation economy government"
        matched: list[str] = []
        impacted: list[dict] = []
        for sector, info in self.sector_map.items():
            if any(kw in text for kw in info["keywords"]):
                matched.append(sector)
                for ticker in info["tickers"]:
                    impacted.append({
                        "entity": ticker,
                        "asset_class": info["asset_class"],
                        "reason": f"{sector} ({source})",
                    })
        # Deduplicate on entity (keep the first).
        seen: set[str] = set()
        unique: list[dict] = []
        for item in impacted:
            if item["entity"] not in seen:
                seen.add(item["entity"])
                unique.append(item)
        confidence = min(1.0, 0.3 + 0.2 * len(matched)) if matched else 0.0
        # Impact vector (category 1): derived from the match.
        # direction: positive if there are growth/positive sectors, else neutral.
        direction = 0.5
        if matched:
            pos_sectors = {"semiconductor", "actuators", "power", "battery",
                           "consumer", "tech", "healthcare", "finance"}
            neg_sectors = {"government", "macro"}
            pos = sum(1 for s in matched if s in pos_sectors)
            neg = sum(1 for s in matched if s in neg_sectors)
            direction = 0.5 + 0.5 * (pos - neg) / max(1, len(matched))
        magnitude = min(1.0, 0.2 + 0.2 * len(matched)) if matched else 0.0
        probability = confidence  # chance the impact occurs ~ confidence
        duration = 0.5 if matched else 0.0  # neutral; the LLM can refine this
        directness = 1.0 if matched else 0.0  # keyword match = directly hit
        novelty = 0.3 if matched else 0.0     # default low; the LLM can raise it
        surprise = 0.0  # default: no market expectation known
        return ImpactResult(
            event=event,
            matched_sectors=matched,
            impacted=unique,
            confidence=round(confidence, 4),
            direction=round(direction, 4),
            magnitude=round(magnitude, 4),
            probability=round(probability, 4),
            duration=round(duration, 4),
            directness=round(directness, 4),
            novelty=round(novelty, 4),
            surprise=round(surprise, 4),
        )

    def to_fusion_inputs(self, result: ImpactResult, sentiment: float = 0.0) -> list[dict]:
        """Convert the impact analysis to fusion inputs per affected entity.

        Each affected entity gets its own fusion input, so the fusion model +
        RL model can decide per instrument. When the entity has its own
        LLM-generated sentiment (e.g. NVDA +0.8 from the impact agent), we use
        THAT per-instrument sentiment instead of the global fallback sentiment.
        """
        if not result.impacted:
            return []
        return [
            {
                "source": "impact",
                "entity_id": item["entity"],
                "asset_class": item["asset_class"],
                "sentiment": round(float(item.get("sentiment", sentiment)), 4),
                "confidence": round(float(item.get("confidence", result.confidence)), 4),
                "reason": item.get("reason", ""),
            }
            for item in result.impacted
        ]


class LLMImpactAgent(ImpactAgent):
    """Impact Agent with optional LLM interpretation.

    Uses your own LLM (via API key/base_url from .env, NOT in GitHub) to
    interpret the event and determine the affected instruments. This catches
    cases that keyword matching misses, e.g.:

        "Trump says Jensen Huang is a good guy, you can trust him"
        -> the LLM understands: this is positive for NVIDIA (NVDA).

    The LLM gets the impact-agent skill (impact_agent/skill/impact-agent-skill.md)
    as the system prompt. Without a key it falls back to keyword matching.
    """

    def __init__(self, sector_map: dict | None = None,
                 api_key: str | None = None,
                 model: str = "stepfun/step-3.7-flash:free",
                 base_url: str = "https://inference-api.nousresearch.com/v1") -> None:
        super().__init__(sector_map)
        self.api_key = api_key
        self.model = model
        self.base_url = base_url

    def _get_key(self) -> str | None:
        if self.api_key:
            return self.api_key
        import os
        return os.environ.get("IMPACT_LLM_API_KEY") or os.environ.get("OPENROUTER_API_KEY")

    def _get_base_url(self) -> str:
        import os
        return os.environ.get("IMPACT_LLM_BASE_URL") or self.base_url

    def analyze(self, event: str, source: str = "news",
                entity_id: str = "") -> ImpactResult:
        """Determine affected instruments, using LLM interpretation if available."""
        # First the fast keyword matching (always).
        base = super().analyze(event, source, entity_id)
        key = self._get_key()
        if not key or not event.strip():
            return base
        # LLM interpretation: ask the model which instruments are hit.
        try:
            llm_impact, llm_meta = self._llm_interpret(event, source, key)
            if llm_impact:
                # Combine: LLM result + keyword matching (deduplicate).
                combined = {i["entity"]: i for i in base.impacted}
                for item in llm_impact:
                    combined[item["entity"]] = item
                base.impacted = list(combined.values())
                base.confidence = round(max(base.confidence, 0.7), 4)
                # Impact vector from the LLM meta (refined vs keyword match).
                for k in ("direction", "magnitude", "probability", "duration",
                          "directness", "novelty", "surprise"):
                    if k in llm_meta:
                        setattr(base, k, round(float(llm_meta[k]), 4))
        except Exception:  # noqa: BLE001
            pass  # fallback to keyword matching
        return base

    def _llm_interpret(self, event: str, source: str, key: str) -> tuple[list[dict], dict]:
        """Ask the LLM (via the skill) which instruments are hit.

        Returns: (instruments, impact-vector meta). The impact-vector meta
        contains the category-1 fields (direction, magnitude, probability,
        duration, directness, novelty, surprise) that the RL model uses as
        its core.
        """
        import json
        import urllib.error
        import urllib.request

        # Load the impact agent skill as the system prompt (if present).
        skill_path = (pathlib.Path(__file__).resolve().parent.parent
                      / "impact_agent" / "skill" / "impact-agent-skill.md")
        system_prompt = "You are a financial impact analyst. Reply with valid JSON only."
        try:
            if skill_path.exists():
                system_prompt = skill_path.read_text()
        except OSError:
            pass

        prompt = (
            f"Event ({source}): {event}\n\n"
            "Reply with a JSON object. List EVERY affected instrument, each "
            "with its OWN sentiment (-1..+1) and confidence (0..1). Respect "
            "the region: if the event names a country/region, include that "
            "region's companies first (e.g. EU AI investment -> MLST, ASML, "
            "STM), then global names that also benefit.\n"
            '{"impact": [{"entity": "NVDA", "asset_class": "stock", '
            '"sentiment": 0.8, "confidence": 0.7, "reason": "..."}], '
            '"meta": {"direction": 0.8, "magnitude": 0.7, "probability": 0.6, '
            '"duration": 0.5, "directness": 0.9, "novelty": 0.4, "surprise": 0.3}}'
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 4000,
            "temperature": 0.1,
        }
        import time
        base = self._get_base_url().rstrip("/")
        req = urllib.request.Request(
            f"{base}/chat/completions", method="POST")
        req.add_header("Authorization", f"Bearer {key}")
        req.add_header("Content-Type", "application/json")
        req.data = json.dumps(payload).encode()
        # Retry on rate-limit (429) and transient errors, with backoff.
        d = {}
        last_err: Exception | None = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    d = json.loads(r.read())
                break
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    last_err = e
                    time.sleep(3 * (attempt + 1))
                    continue
                raise
            except (urllib.error.URLError, OSError, TimeoutError) as e:
                last_err = e
                time.sleep(2 * (attempt + 1))
        if not d and last_err:
            raise last_err
        msg = d["choices"][0]["message"]
        content = (msg.get("content") or msg.get("reasoning") or "").strip()
        # Extract the JSON object from it (the model can put text around it).
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1:
            return [], {}
        try:
            data = json.loads(content[start:end + 1])
        except json.JSONDecodeError:
            return [], {}
        items = data.get("impact", [])
        meta = data.get("meta", {})
        out = []
        for it in items:
            out.append({
                "entity": str(it.get("entity", "")).upper(),
                "asset_class": it.get("asset_class", "stock"),
                "sentiment": float(it.get("sentiment", 0.0)),
                "confidence": float(it.get("confidence", 0.0)),
                "reason": it.get("reason", "llm"),
            })
        return [i for i in out if i["entity"]], meta


def build_impact_agent(config: dict | None = None) -> ImpactAgent:
    """Factory: build the impact agent (keyword matching + optional LLM)."""
    cfg = config or {}
    if cfg.get("use_llm"):
        return LLMImpactAgent(
            model=cfg.get("llm_model", "stepfun/step-3.7-flash:free"))
    return ImpactAgent()
