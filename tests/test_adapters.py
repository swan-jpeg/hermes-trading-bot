"""Tests for typed offline ingestion adapters and timestamp provenance (T05)."""
from __future__ import annotations

import io
import urllib.error
from datetime import UTC, date, datetime
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from hermes_bot.data import DataMode
from hermes_bot.data.adapters import (
    MALFORMED_TIMESTAMP,
    MISSING_PROVIDER_ID,
    NAIVE_TIMESTAMP,
    UNKNOWN_PUBLISHED_TIME,
    AdapterRequest,
    DataInvalidOHLC,
    DataUnavailable,
    ExternalServiceError,
    MarketDataAdapter,
    RawRecord,
    RSSAdapter,
    content_hash,
    document_id_from,
    normalize_documents,
    normalize_market_bars,
    parse_source_datetime,
    with_retry,
)
from hermes_bot.schemas import Adjustment, MarketBar, SourceKind

# ---------------------------------------------------------------------------
# Timestamp parsing
# ---------------------------------------------------------------------------


def test_parse_source_datetime_rfc_aware() -> None:
    dt, flags = parse_source_datetime("Mon, 02 Jan 2023 14:00:00 GMT")
    assert dt is not None
    assert dt.tzinfo is not None
    assert dt.year == 2023 and dt.month == 1 and dt.day == 2
    assert dt.hour == 14
    assert flags == []


def test_parse_source_datetime_iso_aware() -> None:
    dt, flags = parse_source_datetime("2023-01-03T09:30:00+00:00")
    assert dt is not None
    assert dt.tzinfo is not None
    assert flags == []


def test_parse_source_datetime_naive_rejected() -> None:
    dt, flags = parse_source_datetime("2023-01-03T09:30:00")
    assert dt is None
    assert NAIVE_TIMESTAMP in flags


def test_parse_source_datetime_malformed() -> None:
    dt, flags = parse_source_datetime("not-a-date")
    assert dt is None
    assert MALFORMED_TIMESTAMP in flags


def test_parse_source_datetime_missing() -> None:
    dt, flags = parse_source_datetime(None)
    assert dt is None
    assert UNKNOWN_PUBLISHED_TIME in flags


# ---------------------------------------------------------------------------
# Content hash / document ID helpers
# ---------------------------------------------------------------------------


def test_content_hash_is_stable_and_sensitive() -> None:
    h1 = content_hash({"a": 1, "b": 2})
    h2 = content_hash({"b": 2, "a": 1})
    h3 = content_hash({"a": 1, "b": 3})
    assert h1 == h2
    assert h1 != h3
    assert len(h1) == 64


def test_document_id_is_sha256_prefixed() -> None:
    doc_id = document_id_from("https://example.com/x", "a" * 64)
    assert doc_id.startswith("doc_")
    assert len(doc_id) == 4 + 64


# ---------------------------------------------------------------------------
# Market data adapter — offline default
# ---------------------------------------------------------------------------


def test_market_adapter_offline_returns_dates_not_fake_utc() -> None:
    adapter = MarketDataAdapter(mode=DataMode.OFFLINE_MOCK, symbols=["AAPL"])
    request = AdapterRequest(mode=DataMode.OFFLINE_MOCK, source_config={}, max_records=100)
    records = adapter.fetch(request)
    assert len(records) <= 100
    for rec in records:
        sd = rec.payload["session_date"]
        assert isinstance(sd, date)
        assert rec.published_time is None
        assert "daily_session_date_only" in rec.quality_flags


def test_market_adapter_record_limit() -> None:
    adapter = MarketDataAdapter(mode=DataMode.OFFLINE_MOCK, symbols=["AAPL", "MSFT"])
    request = AdapterRequest(mode=DataMode.OFFLINE_MOCK, source_config={}, max_records=2)
    records = adapter.fetch(request)
    assert len(records) == 2


def test_market_adapter_empty_symbols() -> None:
    adapter = MarketDataAdapter(mode=DataMode.OFFLINE_MOCK, symbols=[])
    request = AdapterRequest(mode=DataMode.OFFLINE_MOCK, source_config={}, max_records=100)
    assert adapter.fetch(request) == []


def test_market_adapter_default_makes_no_network_call() -> None:
    with (
        patch("yfinance.download") as mock_yf,
        patch("hermes_bot.data.adapters.urllib.request.urlopen") as mock_urlopen,
    ):
        adapter = MarketDataAdapter()
        adapter.fetch(AdapterRequest(mode=DataMode.OFFLINE_MOCK, source_config={}))
        mock_yf.assert_not_called()
        mock_urlopen.assert_not_called()


def test_market_adapter_network_uses_yfinance() -> None:
    import pandas as pd

    mock_df = pd.DataFrame({
        "Date": ["2023-01-03", "2023-01-04"],
        "Open": [100.0, 101.0],
        "High": [105.0, 106.0],
        "Low": [95.0, 96.0],
        "Close": [102.0, 103.0],
        "Volume": [1_000_000.0, 1_100_000.0],
    })
    with patch("yfinance.download") as mock_download:
        mock_download.return_value = mock_df
        adapter = MarketDataAdapter(mode=DataMode.NETWORK, symbols=["AAPL"])
        records = adapter.fetch(
            AdapterRequest(mode=DataMode.NETWORK, source_config={}, max_records=100)
        )
    assert len(records) == 2
    assert records[0].payload["symbol"] == "AAPL"
    assert records[0].payload["session_date"] == date(2023, 1, 3)


def test_market_adapter_network_exception_is_typed_unavailable() -> None:
    with patch("yfinance.download") as mock_download:
        mock_download.side_effect = ValueError("boom")
        adapter = MarketDataAdapter(mode=DataMode.NETWORK, symbols=["AAPL"])
        with pytest.raises(DataUnavailable):
            adapter.fetch(AdapterRequest(mode=DataMode.NETWORK, source_config={}))


# ---------------------------------------------------------------------------
# Market bar normalizer
# ---------------------------------------------------------------------------


def _mock_market_record(symbol: str, session_date: date, o, h, lo, c, v) -> RawRecord:
    payload = {
        "symbol": symbol,
        "session_date": session_date,
        "interval": "1d",
        "open": o,
        "high": h,
        "low": lo,
        "close": c,
        "volume": v,
        "currency": "USD",
        "exchange_mic": "XNAS",
    }
    now = datetime.now(UTC)
    return RawRecord(
        source_id="mock.market",
        source_uri="mock://market",
        source_kind=SourceKind.MARKET,
        provider_id=f"{symbol}|{session_date.isoformat()}",
        published_time=None,
        ingested_at=now,
        available_at=now,
        content_hash=content_hash(payload),
        license_id="mock",
        payload=payload,
        quality_flags=[],
    )


def test_normalize_market_bars_success() -> None:
    rec = _mock_market_record("AAPL", date(2023, 1, 3), 100.0, 105.0, 95.0, 102.0, 1_000_000.0)
    bars = normalize_market_bars(
        [rec],
        instrument_ids={"AAPL": "ins_00000000-0000-0000-0000-000000000001"},
    )
    assert len(bars) == 1
    bar = bars[0]
    assert isinstance(bar, MarketBar)
    assert bar.session_date == date(2023, 1, 3)
    assert bar.instrument_id == "ins_00000000-0000-0000-0000-000000000001"
    assert bar.adjustment == Adjustment.RAW


def test_normalize_market_bars_rejects_invalid_ohlc() -> None:
    rec = _mock_market_record("AAPL", date(2023, 1, 3), 100.0, 99.0, 95.0, 102.0, 1_000_000.0)
    with pytest.raises(DataInvalidOHLC):
        normalize_market_bars([rec])


def test_market_bar_dedup_key_stable_across_runs() -> None:
    rec1 = _mock_market_record("AAPL", date(2023, 1, 3), 100.0, 105.0, 95.0, 102.0, 1_000_000.0)
    rec2 = _mock_market_record("AAPL", date(2023, 1, 3), 101.0, 106.0, 96.0, 103.0, 1_100_000.0)
    bars = normalize_market_bars([rec1, rec2], instrument_ids={"AAPL": "ins_1"})
    assert bars[0].dedup_key == bars[1].dedup_key


def test_market_bar_dedup_key_changes_with_revision() -> None:
    rec1 = _mock_market_record("AAPL", date(2023, 1, 3), 100.0, 105.0, 95.0, 102.0, 1_000_000.0)
    rec1.metadata["source_revision"] = "v1"
    rec2 = _mock_market_record("AAPL", date(2023, 1, 3), 100.0, 105.0, 95.0, 102.0, 1_000_000.0)
    rec2.metadata["source_revision"] = "v2"
    bars = normalize_market_bars([rec1, rec2], instrument_ids={"AAPL": "ins_1"})
    assert bars[0].dedup_key != bars[1].dedup_key


# ---------------------------------------------------------------------------
# RSS / document adapter
# ---------------------------------------------------------------------------


def test_rss_adapter_offline_returns_mock_documents() -> None:
    adapter = RSSAdapter(
        mode=DataMode.OFFLINE_MOCK,
        source_kind=SourceKind.SPEECH,
        source_id="speech.mock",
    )
    records = adapter.fetch(AdapterRequest(mode=DataMode.OFFLINE_MOCK, max_records=5))
    assert len(records) <= 5
    for rec in records:
        assert rec.source_kind == SourceKind.SPEECH
        assert rec.provider_id is not None
        assert rec.published_time is not None or UNKNOWN_PUBLISHED_TIME in rec.quality_flags


def test_rss_adapter_default_makes_no_network_call() -> None:
    with patch("hermes_bot.data.adapters.urllib.request.urlopen") as mock_urlopen:
        adapter = RSSAdapter(source_kind=SourceKind.ALERT)
        adapter.fetch(AdapterRequest(mode=DataMode.OFFLINE_MOCK, max_records=5))
        mock_urlopen.assert_not_called()


def test_rss_adapter_network_parses_rfc_date_and_malformed_date() -> None:
    xml = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss>
  <channel>
    <item>
      <title>Good date</title>
      <link>https://example.com/good</link>
      <description>Summary</description>
      <pubDate>Mon, 02 Jan 2023 14:00:00 GMT</pubDate>
    </item>
    <item>
      <title>Bad date</title>
      <link>https://example.com/bad</link>
      <description>Summary</description>
      <pubDate>not-a-date</pubDate>
    </item>
  </channel>
</rss>"""
    response = io.BytesIO(xml)

    def fake_urlopen(req, **kwargs):
        return response

    with patch("hermes_bot.data.adapters.urllib.request.urlopen", side_effect=fake_urlopen):
        adapter = RSSAdapter(
            mode=DataMode.NETWORK,
            feeds={"demo": "https://example.com/feed.xml"},
            source_kind=SourceKind.ALERT,
            source_id="alert.rss",
        )
        records = adapter.fetch(AdapterRequest(mode=DataMode.NETWORK, max_records=5))

    assert len(records) == 2
    good = next(r for r in records if r.payload["title"] == "Good date")
    bad = next(r for r in records if r.payload["title"] == "Bad date")
    assert good.published_time is not None
    assert bad.published_time is None
    assert MALFORMED_TIMESTAMP in bad.quality_flags


def test_rss_adapter_network_missing_link_gets_quality_flag() -> None:
    xml = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss>
  <channel>
    <item>
      <title>No link</title>
      <description>Summary</description>
      <pubDate>Mon, 02 Jan 2023 14:00:00 GMT</pubDate>
    </item>
  </channel>
</rss>"""

    def fake_urlopen(req, **kwargs):
        return io.BytesIO(xml)

    with patch("hermes_bot.data.adapters.urllib.request.urlopen", side_effect=fake_urlopen):
        adapter = RSSAdapter(
            mode=DataMode.NETWORK,
            feeds={"demo": "https://example.com/feed.xml"},
            source_kind=SourceKind.ALERT,
        )
        records = adapter.fetch(AdapterRequest(mode=DataMode.NETWORK, max_records=5))

    assert len(records) == 1
    assert records[0].provider_id is None
    assert MISSING_PROVIDER_ID in records[0].quality_flags


def test_rss_adapter_network_empty_feed() -> None:
    xml = b"""<?xml version="1.0" encoding="UTF-8"?><rss><channel></channel></rss>"""

    def fake_urlopen(req, **kwargs):
        return io.BytesIO(xml)

    with patch("hermes_bot.data.adapters.urllib.request.urlopen", side_effect=fake_urlopen):
        adapter = RSSAdapter(
            mode=DataMode.NETWORK,
            feeds={"demo": "https://example.com/feed.xml"},
            source_kind=SourceKind.ALERT,
        )
        assert adapter.fetch(AdapterRequest(mode=DataMode.NETWORK)) == []


# ---------------------------------------------------------------------------
# Document normalizer
# ---------------------------------------------------------------------------


def test_normalize_documents_preserves_source_time_and_ingestion_time() -> None:
    pub = datetime(2023, 1, 2, 14, 0, tzinfo=UTC)
    ing = datetime(2023, 1, 2, 15, 0, tzinfo=UTC)
    payload = {"title": "T", "summary": "S", "language": "en", "document_type": "alert"}
    rec = RawRecord(
        source_id="rss.demo",
        source_uri="https://example.com/item",
        source_kind=SourceKind.ALERT,
        provider_id="https://example.com/item",
        published_time=pub,
        ingested_at=ing,
        available_at=ing,
        content_hash=content_hash(payload),
        license_id="rss",
        payload=payload,
        quality_flags=[],
    )
    docs = normalize_documents([rec])
    assert len(docs) == 1
    assert docs[0].published_time == pub
    assert docs[0].ingested_at == ing
    assert docs[0].document_id.startswith("doc_")


def test_normalize_documents_applies_quality_flags_to_metadata() -> None:
    payload = {"title": "T", "summary": "S"}
    rec = RawRecord(
        source_id="rss.demo",
        source_uri="https://example.com/item",
        source_kind=SourceKind.ALERT,
        provider_id=None,
        published_time=None,
        ingested_at=datetime.now(UTC),
        available_at=datetime.now(UTC),
        content_hash=content_hash(payload),
        license_id="rss",
        payload=payload,
        quality_flags=[UNKNOWN_PUBLISHED_TIME],
    )
    docs = normalize_documents([rec])
    flags = docs[0].metadata.get("quality_flags", [])
    assert UNKNOWN_PUBLISHED_TIME in flags


# ---------------------------------------------------------------------------
# Retry behaviour
# ---------------------------------------------------------------------------


def test_retry_succeeds_after_transient_errors() -> None:
    attempts = {"count": 0}

    def flaky() -> str:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise urllib.error.HTTPError(
                "url", 503, "Service Unavailable", None, io.BytesIO()
            )
        return "ok"

    result = with_retry(flaky)
    assert result == "ok"
    assert attempts["count"] == 3


def test_retry_raises_after_max_transient_attempts() -> None:
    def always_fail() -> str:
        raise urllib.error.HTTPError(
            "url", 503, "Service Unavailable", None, io.BytesIO()
        )

    with pytest.raises(ExternalServiceError):
        with_retry(always_fail)


def test_retry_does_not_retry_non_transient() -> None:
    attempts = {"count": 0}

    def fail_value_error() -> str:
        attempts["count"] += 1
        raise ValueError("boom")

    with pytest.raises(ExternalServiceError):
        with_retry(fail_value_error)
    assert attempts["count"] == 1


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def test_market_bar_schema_rejects_non_positive_price() -> None:
    with pytest.raises(ValidationError):
        MarketBar(
            instrument_id="ins_00000000-0000-0000-0000-000000000001",
            source_id="s",
            interval="1d",
            session_date=date(2023, 1, 3),
            available_at=datetime.now(UTC),
            currency="USD",
            open=100.0,
            high=105.0,
            low=95.0,
            close=0.0,
            volume=1_000_000.0,
        )
