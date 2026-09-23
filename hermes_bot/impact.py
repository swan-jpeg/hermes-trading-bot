"""Impact Agent — koppelt gebeurtenissen aan beïnvloede instrumenten.

De ontbrekende schakel in de keten: bepaalt WELKE instrumenten (stocks,
obligaties, ETF's) geraakt worden door een gebeurtenis (speech, kwartaalrapport,
overheidsuitgave, nieuws). Dit is een PARALLELLE stap vóór het fusion-model:

    webscraping (gebeurtenis)
        ↓
    IMPACT-AGENT  ← bepaalt: "dit raakt NVDA, TSLA, AGG"
        ↓
    per beïnvloede entiteit: fusion-output (sentiment/zekerheid/kwaliteit)
        ↓
    RL-model (per entiteit: welk instrument + gewicht + zekerheid)
        ↓
    risk engine + Monte Carlo → portfolio → executie

Zonder deze agent is het RL-model "blind" — het weet niet welke stocks geraakt
worden door een speech of rapport. Deze agent levert die koppeling.
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
        return ImpactResult(
            event=event,
            matched_sectors=matched,
            impacted=unique,
            confidence=round(confidence, 4),
        )

    def to_fusion_inputs(self, result: ImpactResult, sentiment: float = 0.0) -> list[dict]:
        """Zet de impact-analyse om naar fusion-inputs per beïnvloede entiteit.

        Elke beïnvloede entiteit krijgt een eigen fusion-input, zodat het
        fusion-model + RL-model per instrument kunnen beslissen.
        """
        if not result.impacted:
            return []
        return [
            {
                "source": "impact",
                "entity_id": item["entity"],
                "asset_class": item["asset_class"],
                "sentiment": round(sentiment, 4),
                "confidence": result.confidence,
                "reason": item["reason"],
            }
            for item in result.impacted
        ]


def build_impact_agent(config: dict | None = None) -> ImpactAgent:
    """Factory: bouw de impact-agent (keyword-matching + optionele LLM)."""
    cfg = config or {}
    if cfg.get("use_llm"):
        return LLMImpactAgent(
            model=cfg.get("llm_model", "meta-llama/llama-3.1-8b-instruct:free"))
    return ImpactAgent()


class LLMImpactAgent(ImpactAgent):
    """Impact Agent met optionele LLM-interpretatie.

    Gebruikt een gratis AI-model (via API-key uit .env, NIET in GitHub) om de
    gebeurtenis te interpreteren en beïnvloede instrumenten te bepalen. Dit
    vangt gevallen die keyword-matching mist, bv.:

        "Trump zegt dat Jensen Huang een goede gozer is en je hem kan vertrouwen"
        -> LLM begrijpt: dit is positief voor NVIDIA (NVDA).

    De API-key komt uit de omgeving (IMPACT_LLM_API_KEY / OPENROUTER_API_KEY),
    nooit uit code. Zonder key valt het terug op de keyword-matching.
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
            llm_impact = self._llm_interpret(event, source, key)
            if llm_impact:
                # Combineer: LLM-resultaat + keyword-matching (dedupliceer).
                combined = {i["entity"]: i for i in base.impacted}
                for item in llm_impact:
                    combined[item["entity"]] = item
                base.impacted = list(combined.values())
                base.confidence = round(max(base.confidence, 0.7), 4)
        except Exception:  # noqa: BLE001
            pass  # fallback naar keyword-matching
        return base

    def _llm_interpret(self, event: str, source: str, key: str) -> list[dict]:
        """Vraag het LLM (via de skill) welke instrumenten geraakt worden."""
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

        prompt = f"Gebeurtenis ({source}): {event}"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 300,
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
        # Haal de JSON-array eruit (het model kan er tekst omheen zetten).
        start = content.find("[")
        end = content.rfind("]")
        if start == -1 or end == -1:
            return []
        items = json.loads(content[start:end + 1])
        out = []
        for it in items:
            out.append({
                "entity": str(it.get("entity", "")).upper(),
                "asset_class": it.get("asset_class", "stock"),
                "sentiment": float(it.get("sentiment", 0.0)),
                "reason": it.get("reason", "llm"),
            })
        return [i for i in out if i["entity"]]
