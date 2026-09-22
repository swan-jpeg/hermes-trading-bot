"""LAAG 1-2: gestructureerde webscraping — speech/reports/alerts collectoren.

De architectuur specificeert dat een 24/7-server via gestructureerde
webscraping informatie verzamelt uit speech, reports en alerts. Deze module
implementeert dat met gratis, gestructureerde bronnen:

- SpeechCollector: RSS/nieuwsfeeds van speeches van CEO's en landsleiders
  (transcripts, persconferenties). Bijv. Trump/Fed via nieuws-RSS.
- ReportCollector: bedrijfsrapporten (quarterly reports, overheidsuitgaven).
- AlertCollector: nieuwsberichten en point-loops.

Elke collector is offline-safe: als de netwerk-call faalt, wordt een lege of
offline-demo-reeks teruggegeven (credibility 0) zodat de keten niet crasht.
De echte scraping wordt geactiveerd door enabled=True. Deze server is te zwak
om de zware modellen te draaien, dus de collectoren gebruiken stdlib-only.

Belangrijk: collect() is idempotent + gestructureerd. Elk resultaat is een
dict met de velden die de signalen-laag verwacht.
"""
from __future__ import annotations

import html
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from xml.etree import ElementTree as ET

from hermes_bot.data import BaseCollector, CollectorConfig


def _fetch_rss(url: str, timeout: int = 10) -> list[dict]:
    """Haal een RSS/Atom-feed op en geef de items als dicts.

    Per item: {title, link, summary, published, source}.
    """
    req = urllib.request.Request(url, headers={"User-Agent": "HermesBot/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    root = ET.fromstring(data)
    items = []
    for item in root.iter("item") or root.iter("entry"):
        title = (item.findtext("title") or "").strip()
        link = item.findtext("link")
        if link is None:
            # Atom gebruikt <link href="...">
            link_el = item.find("link")
            link = link_el.get("href") if link_el is not None else ""
        summary = (item.findtext("description") or item.findtext("summary") or "").strip()
        pub = item.findtext("pubDate") or item.findtext("published") or ""
        items.append({
            "title": html.unescape(title),
            "link": link or "",
            "summary": html.unescape(summary),
            "published": pub,
            "source": url,
        })
    return items


# ---------------------------------------------------------------------------
# SPEECH COLLECTOR — speeches van CEO's en landsleiders (via nieuws-RSS)
# ---------------------------------------------------------------------------
SPEECH_FEEDS = {
    "trump": "https://www.whitehouse.gov/news/feed/",
    "ecb": "https://www.ecb.europa.eu/rss/press.html",
    "fed": "https://www.federalreserve.gov/feeds/press_all.xml",
}

TRUMP_KEYWORDS = ["president", "trump", "remarks", "speech", "address", "press conference",
                  "statement", "executive order"]
CEO_KEYWORDS = ["earnings call", "conference call", "ceo", "executive", "guidance",
                "outlook", "quarterly results"]


class SpeechCollector(BaseCollector):
    """Verzamelt speech-transcripts van landsleiders en CEO's.

    Dit levert de RUWE speech-items. De daadwerkelijke audio/video-analyse
    (whisper/DeepFace) gebeurt in de signals-laag zodra een mediabestand
    beschikbaar is. Deze collector levert de tekstuele input.
    """

    name = "speech"

    def __init__(self, cfg: CollectorConfig | None = None, feeds: dict | None = None,
                 max_items: int = 20) -> None:
        super().__init__(cfg)
        # Explicit lege dict is legitiem (geen feeds); None geeft de default.
        self.feeds = SPEECH_FEEDS if feeds is None else feeds
        self.max_items = max_items

    def collect(self) -> list[dict]:
        """Haal recente speech-items op. Offline-safe."""
        if not self.cfg.enabled:
            return []
        records = []
        for speaker, url in self.feeds.items():
            try:
                items = _fetch_rss(url)[:self.max_items]
                for it in items:
                    text = f"{it['title']} {it['summary']}".lower()
                    is_political = any(k in text for k in TRUMP_KEYWORDS)
                    is_ceo = any(k in text for k in CEO_KEYWORDS)
                    records.append({
                        "source": "speech",
                        "speaker": speaker,
                        "headline": it["title"],
                        "body": it["summary"],
                        "url": it["link"],
                        "timestamp": datetime.now(UTC).isoformat(),
                        "is_political": is_political,
                        "is_ceo": is_ceo,
                        "confidence": 0.6 if (is_political or is_ceo) else 0.3,
                    })
            except Exception as e:
                # Offline-safe: log en ga door (credibility blijft laag).
                print(f"[speech] feed {speaker} mislukt: {e}")
        return records


# ---------------------------------------------------------------------------
# REPORT COLLECTOR — bedrijfsrapporten en overheidsuitgaven
# ---------------------------------------------------------------------------
REPORT_FEEDS = {
    "sec": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent",
    "treasury": "https://home.treasury.gov/rss/press-releases.xml",
}


class ReportCollector(BaseCollector):
    """Verzamelt bedrijfsrapporten (quarterly) en overheidsuitgaven.

    Report-type: business | overheidsuitgaven | onafhankelijk.
    """

    name = "report"

    def __init__(self, cfg: CollectorConfig | None = None, feeds: dict | None = None,
                 max_items: int = 20) -> None:
        super().__init__(cfg)
        self.feeds = REPORT_FEEDS if feeds is None else feeds
        self.max_items = max_items

    def collect(self) -> list[dict]:
        """Haal recente rapporten op. Offline-safe."""
        if not self.cfg.enabled:
            return []
        records = []
        for rtype, url in self.feeds.items():
            try:
                items = _fetch_rss(url)[:self.max_items]
                for it in items:
                    records.append({
                        "source": "report",
                        "report_type": "business" if rtype == "sec" else "overheidsuitgaven",
                        "headline": it["title"],
                        "body": it["summary"],
                        "url": it["link"],
                        "timestamp": datetime.now(UTC).isoformat(),
                        "confidence": 0.6,
                    })
            except Exception as e:
                print(f"[report] feed {rtype} mislukt: {e}")
        return records


# ---------------------------------------------------------------------------
# ALERT COLLECTOR — nieuwsberichten en point-loops
# ---------------------------------------------------------------------------
ALERT_FEEDS = {
    "news": "https://feeds.bbci.co.uk/news/business/rss.xml",
    "markets": "https://feeds.marketwatch.com/marketwatch/topstories/",
}


class AlertCollector(BaseCollector):
    """Verzamelt nieuws-alerts en point-loops voor regionale scores."""

    name = "alert"

    def __init__(self, cfg: CollectorConfig | None = None, feeds: dict | None = None,
                 max_items: int = 20) -> None:
        super().__init__(cfg)
        self.feeds = ALERT_FEEDS if feeds is None else feeds
        self.max_items = max_items

    def collect(self) -> list[dict]:
        if not self.cfg.enabled:
            return []
        records = []
        for kind, url in self.feeds.items():
            try:
                items = _fetch_rss(url)[:self.max_items]
                for it in items:
                    records.append({
                        "source": "alert",
                        "kind": "news" if kind == "news" else "point_loop",
                        "headline": it["title"],
                        "body": it["summary"],
                        "url": it["link"],
                        "timestamp": datetime.now(UTC).isoformat(),
                        "confidence": 0.5,
                    })
            except Exception as e:
                print(f"[alert] feed {kind} mislukt: {e}")
        return records


# ---------------------------------------------------------------------------
# WEB SCRAPER — orchestreert alle collectoren
# ---------------------------------------------------------------------------
class WebScraper:
    """Centrale webscraper die speech/reports/alerts verzamelt."""

    def __init__(self, config: dict | None = None) -> None:
        self.cfg = config or {}
        r = self.cfg.get("routes", {})
        master = self.cfg.get("enabled", False)
        self.speech = SpeechCollector(CollectorConfig(
            name="speech", enabled=r.get("speech", {}).get("enabled", master),
            interval_sec=r.get("speech", {}).get("interval_sec", 3600)))
        self.report = ReportCollector(CollectorConfig(
            name="report", enabled=r.get("reports", {}).get("enabled", master),
            interval_sec=r.get("reports", {}).get("interval_sec", 86400)))
        self.alert = AlertCollector(CollectorConfig(
            name="alert", enabled=r.get("alerts", {}).get("enabled", master),
            interval_sec=r.get("alerts", {}).get("interval_sec", 600)))

    @classmethod
    def from_config_file(cls) -> WebScraper:
        """Bouw een WebScraper uit config.yaml (webscraping sectie)."""
        from hermes_bot.config import load_config
        cfg = load_config()
        return cls(cfg.get("webscraping", {}))

    def collect_all(self) -> dict[str, list[dict]]:
        """Verzamel alle routes in één keer."""
        return {
            "speech": self.speech.collect(),
            "reports": self.report.collect(),
            "alerts": self.alert.collect(),
        }


def demo_speech_records() -> list[dict]:
    """Offline-demo-items om de keten te testen zonder netwerk."""
    return [
        {
            "source": "speech", "speaker": "trump",
            "headline": "President Trump: remarks on infrastructure spending",
            "body": "We are investing heavily in domestic manufacturing and grid infrastructure. "
                    "Semiconductor and actuator supply must be built in America.",
            "url": "https://example.com/trump/remarks",
            "timestamp": datetime.now(UTC).isoformat(),
            "is_political": True, "is_ceo": False, "confidence": 0.8,
        },
        {
            "source": "speech", "speaker": "ceo",
            "headline": "SEMI CEO earnings call: capacity constraints",
            "body": "Our semiconductor equipment order backlog is at record levels. "
                    "Lead times for actuators and power components have doubled.",
            "url": "https://example.com/ceo/earnings",
            "timestamp": datetime.now(UTC).isoformat(),
            "is_political": False, "is_ceo": True, "confidence": 0.9,
        },
    ]


def demo_report_records() -> list[dict]:
    return [
        {
            "source": "report", "report_type": "business",
            "headline": "NVDA quarterly results beat expectations",
            "body": "Revenue grew 65% YoY on AI chip demand. Gross margin expanded. "
                    "Management raised full-year guidance citing supply chain bottlenecks.",
            "url": "https://example.com/nvda/q",
            "timestamp": datetime.now(UTC).isoformat(), "confidence": 0.8,
        },
        {
            "source": "report", "report_type": "overheidsuitgaven",
            "headline": "US government infrastructure spending bill",
            "body": "New federal budget allocates $50B to semiconductor and grid infrastructure, "
                    "targeting domestic manufacturing capacity.",
            "url": "https://example.com/gov/infra",
            "timestamp": datetime.now(UTC).isoformat(), "confidence": 0.7,
        },
    ]


def demo_alert_records() -> list[dict]:
    return [
        {
            "source": "alert", "kind": "news",
            "headline": "Supply chain bottleneck intensifies",
            "body": "Actuator and power semiconductor shortages worsen as demand "
                     "outpaces capacity.",
            "url": "https://example.com/news/bottleneck",
            "timestamp": datetime.now(UTC).isoformat(),
            "confidence": 0.6,
        },
    ]


def _sentiment_from_text(text: str) -> float:
    """Eenvoudige lexicon-sentiment (-1..1). Vervangt geen FinBERT; prima voor offline-keten."""
    pos = ["growth", "beat", "expand", "invest", "record", "guidance", "demand", "strong",
           "improve", "opportunity"]
    neg = ["shortage", "delay", "constraint", "worsen", "decline", "risk", "weak",
           "bottleneck", "cut"]
    t = text.lower()
    p = sum(1 for w in pos if w in t)
    n = sum(1 for w in neg if w in t)
    total = p + n
    if total == 0:
        return 0.0
    return float((p - n) / total)


# ---------------------------------------------------------------------------
# CONVERSIE: ruwe webitems -> signalen voor het fusion model
# ---------------------------------------------------------------------------
def web_records_to_inputs(
    speech: list[dict], reports: list[dict], alerts: list[dict], entity_id: str = "market"
) -> list[dict]:
    """Converteer ruwe webitems naar fusie-inputs {source, sentiment, confidence}.

    Dit is de brug: data/ (webscraping) -> signals/ (verwerking) -> fusion/.
    Voor speech wordt de audio/video-analyse (whisper/DeepFace) normaal
    toegepast; hier wordt alleen tekstuele sentiment toegepast (offline-keten).
    """
    inputs = []
    for it in speech:
        sent = _sentiment_from_text(f"{it.get('headline','')} {it.get('body','')}")
        inputs.append({
            "source": "speech",
            "entity_id": it.get("speaker", entity_id),
            "sentiment": sent,
            "confidence": it.get("confidence", 0.5),
        })
    for it in reports:
        sent = _sentiment_from_text(f"{it.get('headline','')} {it.get('body','')}")
        inputs.append({
            "source": "report",
            "entity_id": entity_id,
            "sentiment": sent,
            "confidence": it.get("confidence", 0.5),
        })
    for it in alerts:
        sent = _sentiment_from_text(f"{it.get('headline','')} {it.get('body','')}")
        inputs.append({
            "source": "alert",
            "entity_id": entity_id,
            "sentiment": sent,
            "confidence": it.get("confidence", 0.5),
        })
    return inputs


def scrape_and_fuse(web_scraper: WebScraper, entity_id: str = "market") -> dict:
    """End-to-end: scrape -> convert -> fuse. Geeft FusionSignal-dict.

    Gebruikt de echte collectoren als enabled, anders demo-data.
    """
    from hermes_bot.data.scraper import web_records_to_inputs
    from hermes_bot.fusion import WeightedFusion

    data = web_scraper.collect_all()
    use_demo = not (web_scraper.speech.cfg.enabled or web_scraper.report.cfg.enabled
                    or web_scraper.alert.cfg.enabled)
    if use_demo:
        speech, reports, alerts = demo_speech_records(), demo_report_records(), demo_alert_records()
    else:
        speech, reports, alerts = data["speech"], data["reports"], data["alerts"]

    inputs = web_records_to_inputs(speech, reports, alerts, entity_id)
    if not inputs:
        return {"error": "geen web-inputs", "inputs": []}
    fusion = WeightedFusion()
    result = fusion.fuse(inputs)
    return {"fusion": result.model_dump(), "n_inputs": len(inputs)}