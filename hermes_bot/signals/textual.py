"""LAAG 3-4: reports, alerts & regionale scores. [[04]]"""
from __future__ import annotations

from hermes_bot.schemas import BaseSignal, SourceKind


class ReportSignal(BaseSignal):
    source: SourceKind = SourceKind.REPORT
    report_type: str = "business"  # business | overheidsuitgaven | onafhankelijk
    summary: str = ""
    extracted_topics: list[str] = []


class AlertSignal(BaseSignal):
    source: SourceKind = SourceKind.ALERT
    kind: str = "news"  # news | point_loop
    headline: str = ""
    body: str = ""
    sentiment: float = 0.0  # -1..1
    credibility: float = 0.5  # 0..1 bronweging


class RegionalScores(BaseSignal):
    source: SourceKind = SourceKind.REGIONAL
    region: str = ""
    scores: dict[str, float] = {
        "veiligheid": 0.0,
        "tevredenheid": 0.0,
        "sociale_zekerheid": 0.0,
        "bedrijfseconomische_veranderingen": 0.0,
    }
