"""Impact Agent — koppelt gebeurtenissen aan beïnvloede instrumenten.

De ontbrekende schakel in de keten: bepaalt WELKE instrumenten (stocks,
obligaties, ETF's) geraakt worden door een gebeurtenis (speech, kwartaalrapport,
overheidsuitgave, nieuws). Dit is een PARALLELLE stap vóór het fusion-model:

    webscraping (gebeurtenis)
        ↓
    IMPACT-AGENT  ← bepaalt: "dit raakt NVDA, TSLA, AGG"
        ↓
    per beïnvloede entiteit: vector {entity, asset_class, sentiment, confidence}
        ↓
    RL-model (per entiteit: welk instrument + gewicht + zekerheid)
        ↓
    risk engine + Monte Carlo → portfolio → executie

Zonder deze agent is het RL-model "blind" — het weet niet welke stocks geraakt
worden door een speech of rapport. Deze agent levert die koppeling.

Twee modi:
- ImpactAgent: keyword/sector-matching (snel, gratis, offline, altijd).
- LLMImpactAgent: gebruikt een eigen LLM met de impact-agent-skill
  (impact_agent/skill/impact-agent-skill.md) om de context te begrijpen.
  De API-key/base_url komen uit .env, nooit uit code.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass, field

# Sector/onderwerp -> beïnvloede instrumenten (tickers + assetklasse).
# Uitbreidbaar: voeg keywords toe per sector.
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
    """Output van de impact-agent: welke instrumenten worden geraakt."""

    event: str
    matched_sectors: list[str] = field(default_factory=list)
    impacted: list[dict] = field(default_factory=list)  # [{entity, asset_class, reason}]
    confidence: float = 0.0
    # Impact-vector (categorie 1 van de RL-inputs): per analyse, genormaliseerd.
    direction: float = 0.5       # 0=negatief, 1=positief
    magnitude: float = 0.0        # hoe groot de verwachte impact
    probability: float = 0.0     # kans dat de impact optreedt
    duration: float = 0.5        # hoe lang de impact blijft
    directness: float = 0.0      # 1=direct geraakt, 0=indirect
    novelty: float = 0.0         # hoe nieuw/onverwacht
    surprise: float = 0.0        # afwijking van wat al verwacht werd


class ImpactAgent:
    """Koppelt een gebeurtenis aan beïnvloede instrumenten via keyword-matching."""

    def __init__(self, sector_map: dict | None = None) -> None:
        self.sector_map = sector_map or _SECTOR_MAP

    def analyze(self, event: str, source: str = "news",
                entity_id: str = "") -> ImpactResult:
        """Bepaal welke instrumenten door deze gebeurtenis worden geraakt.

        event: de tekst van de gebeurtenis (speech, rapport, overheidsuitgave).
        source: speech | report | alert | government.
        entity_id: optionele entiteit (bv. "trump") die ook als hint dient.
        """
        text = (event + " " + entity_id).lower()
        # Speech van een wereldleider (trump, biden, putin, ...) -> macro-impact.
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
        # Dedupliceer op entity (behoud eerste).
        seen: set[str] = set()
        unique: list[dict] = []
        for item in impacted:
            if item["entity"] not in seen:
                seen.add(item["entity"])
                unique.append(item)
        confidence = min(1.0, 0.3 + 0.2 * len(matched)) if matched else 0.0
        # Impact-vector (categorie 1): afgeleid uit de match.
        # direction: positief als er groei/positieve sectoren zijn, anders neutraal.
        direction = 0.5
        if matched:
            pos_sectors = {"semiconductor", "actuators", "power", "battery",
                           "consumer", "tech", "healthcare", "finance"}
            neg_sectors = {"government", "macro"}
            pos = sum(1 for s in matched if s in pos_sectors)
            neg = sum(1 for s in matched if s in neg_sectors)
            direction = 0.5 + 0.5 * (pos - neg) / max(1, len(matched))
        magnitude = min(1.0, 0.2 + 0.2 * len(matched)) if matched else 0.0
        probability = confidence  # kans dat de impact optreedt ~ zekerheid
        duration = 0.5 if matched else 0.0  # neutraal; LLM kan dit verfijnen
        directness = 1.0 if matched else 0.0  # keyword-match = direct geraakt
        novelty = 0.3 if matched else 0.0     # default laag; LLM kan verhogen
        surprise = 0.0  # default: geen marktverwachting bekend
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
        """Zet de impact-analyse om naar fusion-inputs per beïnvloede entiteit.

        Elke beïnvloede entiteit krijgt een eigen fusion-input, zodat het
        fusion-model + RL-model per instrument kunnen beslissen. Wanneer de
        entiteit een eigen LLM-gegenereerd sentiment heeft (bv. NVDA +0.8 van
        de impact-agent), gebruiken we DAT per-instrument sentiment in plaats
        van het globale fallback-sentiment.
        """
        if not result.impacted:
            return []
        return [
            {
                "source": "impact",
                "entity_id": item["entity"],
                "asset_class": item["asset_class"],
                "sentiment": round(float(item.get("sentiment", sentiment)), 4),
                "confidence": result.confidence,
                "reason": item.get("reason", ""),
            }
            for item in result.impacted
        ]


class LLMImpactAgent(ImpactAgent):
    """Impact Agent met optionele LLM-interpretatie.

    Gebruikt een eigen LLM (via API-key/base_url uit .env, NIET in GitHub) om de
    gebeurtenis te interpreteren en beïnvloede instrumenten te bepalen. Dit
    vangt gevallen die keyword-matching mist, bv.:

        "Trump zegt dat Jensen Huang een goede gozer is en je hem kan vertrouwen"
        -> LLM begrijpt: dit is positief voor NVIDIA (NVDA).

    De LLM krijgt de impact-agent-skill (impact_agent/skill/impact-agent-skill.md)
    als system-prompt. Zonder key valt het terug op de keyword-matching.
    """

    def __init__(self, sector_map: dict | None = None,
                 api_key: str | None = None,
                 model: str = "meta-llama/llama-3.1-8b-instruct:free",
                 base_url: str = "https://openrouter.ai/api/v1") -> None:
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
        """Bepaal beïnvloede instrumenten, met LLM-interpretatie als beschikbaar."""
        # Eerst de snelle keyword-matching (altijd).
        base = super().analyze(event, source, entity_id)
        key = self._get_key()
        if not key or not event.strip():
            return base
        # LLM-interpretatie: vraag het model welke instrumenten geraakt worden.
        try:
            llm_impact, llm_meta = self._llm_interpret(event, source, key)
            if llm_impact:
                # Combineer: LLM-resultaat + keyword-matching (dedupliceer).
                combined = {i["entity"]: i for i in base.impacted}
                for item in llm_impact:
                    combined[item["entity"]] = item
                base.impacted = list(combined.values())
                base.confidence = round(max(base.confidence, 0.7), 4)
                # Impact-vector uit de LLM-meta (verfijnd t.o.v. keyword-match).
                for k in ("direction", "magnitude", "probability", "duration",
                          "directness", "novelty", "surprise"):
                    if k in llm_meta:
                        setattr(base, k, round(float(llm_meta[k]), 4))
        except Exception:  # noqa: BLE001
            pass  # fallback naar keyword-matching
        return base

    def _llm_interpret(self, event: str, source: str, key: str) -> tuple[list[dict], dict]:
        """Vraag het LLM (via de skill) welke instrumenten geraakt worden.

        Returns: (instrumenten, impact-vector-meta). De impact-vector-meta
        bevat de categorie-1 velden (direction, magnitude, probability,
        duration, directness, novelty, surprise) die het RL-model als kern
        gebruikt.
        """
        import json
        import urllib.error
        import urllib.request

        # Laad de impact-agent skill als system-prompt (indien aanwezig).
        skill_path = (pathlib.Path(__file__).resolve().parent.parent
                      / "impact_agent" / "skill" / "impact-agent-skill.md")
        system_prompt = "Je bent een financieel impact-analist. Antwoord alleen met geldige JSON."
        try:
            if skill_path.exists():
                system_prompt = skill_path.read_text()
        except OSError:
            pass

        prompt = (
            f"Gebeurtenis ({source}): {event}\n\n"
            "Antwoord met een JSON-object:\n"
            '{"impact": [{"entity": "NVDA", "asset_class": "stock", '
            '"sentiment": 0.8, "reason": "..."}], '
            '"meta": {"direction": 0.8, "magnitude": 0.7, "probability": 0.6, '
            '"duration": 0.5, "directness": 0.9, "novelty": 0.4, "surprise": 0.3}}'
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 400,
            "temperature": 0.1,
        }
        base = self._get_base_url().rstrip("/")
        req = urllib.request.Request(
            f"{base}/chat/completions", method="POST")
        req.add_header("Authorization", f"Bearer {key}")
        req.add_header("Content-Type", "application/json")
        req.data = json.dumps(payload).encode()
        with urllib.request.urlopen(req, timeout=30) as r:
            d = json.loads(r.read())
        content = d["choices"][0]["message"]["content"].strip()
        # Haal het JSON-object eruit (het model kan er tekst omheen zetten).
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
                "reason": it.get("reason", "llm"),
            })
        return [i for i in out if i["entity"]], meta


def build_impact_agent(config: dict | None = None) -> ImpactAgent:
    """Factory: bouw de impact-agent (keyword-matching + optionele LLM)."""
    cfg = config or {}
    if cfg.get("use_llm"):
        return LLMImpactAgent(
            model=cfg.get("llm_model", "meta-llama/llama-3.1-8b-instruct:free"))
    return ImpactAgent()
