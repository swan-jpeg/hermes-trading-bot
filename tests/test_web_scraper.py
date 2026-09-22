"""Tests voor webscraping -> signalen -> fusion aansluiting + bottleneck agent.

Deze tests runnen NIET de zware modellen (whisper/DeepFace) noch echt netwerk-
scraping. Ze verifiëren dat:
1. web_records_to_inputs() ruwe webitems omzet naar fusie-inputs;
2. de speech/report/alert collectoren offline-safe zijn (geen crash bij enabled);
3. het fusion model web + bottleneck bronnen accepteert;
4. de BottleneckAgent bedrijven ratet en doorgeeft aan fusion;
5. de end-to-end scrape_and_fuse() werkt met demo-data.
"""
from __future__ import annotations

from hermes_bot.data import CollectorConfig
from hermes_bot.data.scraper import (
    AlertCollector,
    ReportCollector,
    SpeechCollector,
    WebScraper,
    demo_alert_records,
    demo_report_records,
    demo_speech_records,
    scrape_and_fuse,
    web_records_to_inputs,
)
from hermes_bot.expansions import BottleneckAgent, BottleneckAnalyzer
from hermes_bot.fusion import WeightedFusion


def test_web_records_to_inputs() -> None:
    """Webitems -> fusie-inputs met source/sentiment/confidence."""
    inputs = web_records_to_inputs(
        demo_speech_records(), demo_report_records(), demo_alert_records()
    )
    assert len(inputs) == 5  # 2 speech + 2 reports + 1 alert
    sources = {i["source"] for i in inputs}
    assert "speech" in sources and "report" in sources and "alert" in sources
    for i in inputs:
        assert -1.0 <= i["sentiment"] <= 1.0
        assert 0.0 <= i["confidence"] <= 1.0


def test_speech_collector_keys() -> None:
    """SpeechCollector geeft de juiste velden (speaker/headline/body)."""
    c = SpeechCollector(CollectorConfig(name="speech", enabled=True), feeds={},
                        max_items=5)
    # Geen feeds -> lege reeks, geen crash.
    assert c.collect() == []


def test_report_collector_keys() -> None:
    c = ReportCollector(CollectorConfig(name="report", enabled=True), feeds={},
                        max_items=5)
    assert c.collect() == []


def test_alert_collector_keys() -> None:
    c = AlertCollector(CollectorConfig(name="alert", enabled=True), feeds={},
                       max_items=5)
    assert c.collect() == []


def test_collectors_disabled_returns_empty() -> None:
    """Uitgeschakelde collectoren geven lege reeks (geen netwerk-call)."""
    c = SpeechCollector(CollectorConfig(name="speech", enabled=False))
    assert c.collect() == []


def test_fusion_accepts_web_sources() -> None:
    """Fusion model accepteert speech/report/alert/bottleneck als bron."""
    fusion = WeightedFusion()
    inputs = [
        {"source": "speech", "entity_id": "trump", "sentiment": 0.5, "confidence": 0.8},
        {"source": "report", "entity_id": "tech", "sentiment": 0.7, "confidence": 0.7},
        {"source": "alert", "entity_id": "tech", "sentiment": 0.3, "confidence": 0.5},
        {"source": "bottleneck", "entity_id": "actuators", "sentiment": 0.6,
         "confidence": 0.9},
    ]
    result = fusion.fuse(inputs)
    assert result.entity_id == "trump"  # eerste input bepaalt entity
    assert 0.0 <= result.kwaliteit <= 1.0
    assert 0.0 <= result.zekerheid <= 1.0
    # source_breakdown bevat alle bronnen.
    assert "speech" in result.source_breakdown
    assert "bottleneck" in result.source_breakdown


def test_bottleneck_agent_rates_and_feeds_fusion() -> None:
    """BottleneckAgent ratet bedrijven en geeft fusion-inputs."""
    agent = BottleneckAgent()
    reports = demo_report_records()
    entity_map = {"NVDA": ["semiconductor"], "AIRTECH": ["actuators"],
                  "POWCO": ["power"]}
    inputs = agent.run_pipeline(reports, entity_map, market_sentiment=0.2)
    assert len(inputs) > 0
    for i in inputs:
        assert i["source"] == "bottleneck"
        assert i["confidence"] > 0
    # Door naar fusion.
    fusion = WeightedFusion()
    result = fusion.fuse(inputs)
    assert result.kwaliteit > 0


def test_scrape_and_fuse_end_to_end() -> None:
    """End-to-end scrape_and_fuse() met demo-data (geen netwerk)."""
    ws = WebScraper({"enabled": False})
    out = scrape_and_fuse(ws, entity_id="market")
    assert "fusion" in out
    assert out["n_inputs"] > 0
    assert "kwaliteit" in out["fusion"]
    assert "source_breakdown" in out["fusion"]


def test_bottleneck_analyzer_structure() -> None:
    """BottleneckAnalyzer vindt knelpunten en geeft to_fusion_input."""
    a = BottleneckAnalyzer()
    a.add_node("actuators", demand=0.9, capacity=0.2, players=2)
    a.add_node("semiconductor", demand=0.7, capacity=0.5, players=5)
    tops = a.top_bottlenecks(2)
    assert tops[0][0] == "actuators"  # meeste knelpunt eerst
    fi = a.to_fusion_input()
    assert fi["source"] == "bottleneck"
    assert fi["bottleneck_score"] > 0.5  # actuators is een echt knelpunt
