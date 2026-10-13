"""Typed offline-first ingestion adapters and timestamp provenance (T05).

The default execution mode is ``OFFLINE_MOCK``.  Network adapters are isolated
behind the explicit ``NETWORK`` mode and are not instantiated in offline runs.
"""
from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from email.utils import parsedate_to_datetime
from typing import Any, Protocol
from xml.etree import ElementTree as ET

from hermes_bot.data import DataMode
from hermes_bot.schemas import Adjustment, Document, MarketBar, SourceKind

# ---------------------------------------------------------------------------
# Errors and quality flags
# ---------------------------------------------------------------------------


class DataUnavailable(Exception):
    """A required data source is absent, stale or produced invalid records."""


class ExternalServiceError(Exception):
    """A network/external service call failed after bounded retry."""


# Quality flags are plain strings so tests and downstream code can extend them.
UNKNOWN_PUBLISHED_TIME = "unknown_published_time"
MISSING_PROVIDER_ID = "missing_provider_id"
MALFORMED_TIMESTAMP = "malformed_timestamp"
NAIVE_TIMESTAMP = "naive_timestamp"
DAILY_SESSION_DATE_ONLY = "daily_session_date_only"
INVALID_OHLC = "invalid_ohlc"
MISSING_PRICE_FIELD = "missing_price_field"


# ---------------------------------------------------------------------------
# Mode / request / raw record contracts
# ---------------------------------------------------------------------------


@dataclass
class AdapterRequest:
    """Per-fetch request handed to every adapter."""

    mode: DataMode
    source_config: dict[str, Any] = field(default_factory=dict)
    max_records: int = 100
    timeout: int = 10
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class RawRecord:
    """Untyped source payload with provenance metadata.

    ``provider_id`` should be a stable identifier supplied by the publisher
    (e.g. a permalink, ticker+session date, or provider story ID).  When no
    stable provider ID exists, the dedup key falls back to ``content_hash``.
    """

    source_id: str
    source_uri: str
    source_kind: SourceKind
    provider_id: str | None
    published_time: datetime | None
    ingested_at: datetime
    available_at: datetime
    content_hash: str
    license_id: str | None
    payload: dict[str, Any]
    quality_flags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def dedup_key(self) -> str:
        """Deterministic dedup key: source ID + provider ID or content hash."""
        stable = self.provider_id or self.content_hash
        return f"{self.source_id}::{stable}"


class Adapter(Protocol):
    """Protocol for all ingestion adapters."""

    def fetch(self, request: AdapterRequest) -> list[RawRecord]:
        """Return raw records for the given request."""
        ...


# ---------------------------------------------------------------------------
# Helpers: content hash, timestamp parsing, retry
# ---------------------------------------------------------------------------


def content_hash(payload: dict[str, Any]) -> str:
    """SHA-256 over a canonical JSON serialization of ``payload``."""
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def document_id_from(source_uri: str, raw_sha256: str) -> str:
    """Stable document ID based on normalized source URI plus content hash (§4.1)."""
    text = f"{source_uri.strip().lower()}|{raw_sha256}"
    return "doc_" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_source_datetime(value: str | None) -> tuple[datetime | None, list[str]]:
    """Parse an RSS/market source timestamp into timezone-aware UTC.

    Returns ``(datetime, flags)``.  Missing, malformed or naive values stay
    ``null`` with an explicit quality flag — ingestion time is never substituted.
    """
    flags: list[str] = []
    if value is None or not str(value).strip():
        return None, [UNKNOWN_PUBLISHED_TIME]

    value = str(value).strip()
    parsed: datetime | None = None

    # RFC 822 / RSS pubDate first (e.g. "Mon, 02 Jan 2023 12:00:00 GMT").
    try:
        parsed = parsedate_to_datetime(value)
    except Exception:
        parsed = None

    # ISO-8601 fallback.
    if parsed is None:
        try:
            normalized = value.replace("Z", "+00:00")
            parsed = datetime.fromisoformat(normalized)
        except Exception:
            parsed = None

    if parsed is None:
        return None, [MALFORMED_TIMESTAMP]

    if parsed.tzinfo is None:
        # Publisher did not state a timezone; guessing violates the stop-rule.
        return None, [NAIVE_TIMESTAMP]

    # Normalize to UTC without changing the instant.
    utc = parsed.astimezone(UTC)
    return utc, flags


@dataclass
class RetryConfig:
    """Bounded retry configuration for transient network errors."""

    max_attempts: int = 3
    backoff_seconds: float = 1.0


def is_transient_network_error(exc: Exception) -> bool:
    """Return True when ``exc`` is worth retrying."""
    if isinstance(exc, TimeoutError):
        return True
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in (429, 500, 502, 503, 504)
    if isinstance(exc, urllib.error.URLError):
        return True
    return False


def with_retry(
    operation: Any,
    *,
    is_transient: Any = is_transient_network_error,
    config: RetryConfig | None = None,
) -> Any:
    """Run ``operation`` with bounded retry on transient errors."""
    cfg = config or RetryConfig()
    last_exc: Exception | None = None
    for attempt in range(1, cfg.max_attempts + 1):
        try:
            return operation()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if not is_transient(exc) or attempt == cfg.max_attempts:
                break
            time.sleep(cfg.backoff_seconds * (2 ** (attempt - 1)))
    raise ExternalServiceError(
        f"Operation failed after {cfg.max_attempts} attempt(s)"
    ) from last_exc


# ---------------------------------------------------------------------------
# Normalizers
# ---------------------------------------------------------------------------


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _as_float(value: Any, field_name: str) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise DataUnavailable(f"Non-numeric {field_name}: {value!r}") from exc


def normalize_market_bars(
    records: list[RawRecord],
    *,
    available_at: datetime | None = None,
    instrument_ids: dict[str, str] | None = None,
    currency: str = "USD",
    exchange_mic: str | None = None,
    adjustment: Adjustment = Adjustment.RAW,
) -> list[MarketBar]:
    """Convert market RawRecords into canonical ``MarketBar`` objects.

    ``instrument_ids`` maps a symbol (from ``record.payload["symbol"]``) to a
    canonical ``ins_<UUID>`` instrument ID.
    """
    bars: list[MarketBar] = []
    ids = instrument_ids or {}
    now = available_at or _utc_now()

    for rec in records:
        p = rec.payload
        symbol = str(p.get("symbol", ""))
        if not symbol:
            raise DataUnavailable("Market record missing symbol")

        instrument_id = ids.get(symbol, symbol)
        session_date = p.get("session_date") or p.get("date") or p.get("timestamp")
        if isinstance(session_date, datetime):
            session_date = session_date.date()
        if not isinstance(session_date, date):
            raise DataUnavailable(f"Market record missing session_date: {rec.dedup_key}")

        o = _as_float(p.get("open"), "open")
        h = _as_float(p.get("high"), "high")
        lo = _as_float(p.get("low"), "low")
        c = _as_float(p.get("close"), "close")
        v = _as_float(p.get("volume"), "volume")

        if not all(map(lambda x: x > 0 and x < float("inf"), [o, h, lo, c])):
            raise DataUnavailable(f"Non-finite/positive price in {rec.dedup_key}")
        if v < 0 or v == float("inf"):
            raise DataUnavailable(f"Invalid volume in {rec.dedup_key}")
        if not (lo <= min(o, c) <= max(o, c) <= h):
            raise DataInvalidOHLC(rec)

        quality = list(rec.quality_flags)
        if rec.published_time is None:
            quality.append(DAILY_SESSION_DATE_ONLY)

        bars.append(
            MarketBar(
                instrument_id=instrument_id,
                source_id=rec.source_id,
                interval=str(p.get("interval", "1d")),
                session_date=session_date,
                exchange_mic=p.get("exchange_mic") or exchange_mic,
                interval_start=p.get("interval_start"),
                interval_end=p.get("interval_end"),
                available_at=rec.available_at or now,
                currency=p.get("currency") or currency,
                open=o,
                high=h,
                low=lo,
                close=c,
                volume=v,
                adjustment=p.get("adjustment", adjustment),
                source_revision=rec.metadata.get("source_revision"),
                quality_flags=quality,
            )
        )
    return bars


class DataInvalidOHLC(DataUnavailable):
    """Raised when OHLC invariants are violated."""

    def __init__(self, record: RawRecord) -> None:
        super().__init__(f"OHLC invariant violated: {record.dedup_key}")
        self.record = record


def normalize_documents(records: list[RawRecord]) -> list[Document]:
    """Convert RSS/document RawRecords into canonical ``Document`` metadata."""
    docs: list[Document] = []
    for rec in records:
        raw_sha256 = rec.content_hash
        document_id = document_id_from(rec.source_uri, raw_sha256)
        metadata = dict(rec.metadata)
        if rec.quality_flags:
            metadata["quality_flags"] = list(rec.quality_flags)

        docs.append(
            Document(
                document_id=document_id,
                source_id=rec.source_id,
                source_uri=rec.source_uri,
                raw_sha256=raw_sha256,
                language=rec.payload.get("language"),
                document_type=rec.payload.get("document_type", rec.source_kind.value),
                published_time=rec.published_time,
                ingested_at=rec.ingested_at,
                available_at=rec.available_at,
                license_id=rec.license_id,
                raw_object_path=rec.metadata.get("raw_object_path"),
                metadata=metadata,
            )
        )
    return docs


# ---------------------------------------------------------------------------
# Market data adapter
# ---------------------------------------------------------------------------


class MarketDataAdapter:
    """Adapter for daily OHLCV market bars.

    ``OFFLINE_MOCK`` returns deterministic synthetic bars and never imports or
    calls ``yfinance``.  ``NETWORK`` is the isolated legacy yfinance path.
    """

    def __init__(
        self,
        mode: DataMode = DataMode.OFFLINE_MOCK,
        symbols: list[str] | None = None,
        currency: str = "USD",
        exchange_mic: str | None = "XNAS",
    ) -> None:
        self.mode = mode
        self.symbols = ["AAPL", "MSFT", "SPY"] if symbols is None else symbols
        self.currency = currency
        self.exchange_mic = exchange_mic

    def fetch(self, request: AdapterRequest) -> list[RawRecord]:
        """Fetch raw market records."""
        symbols = request.source_config.get("symbols") or self.symbols
        max_records = request.max_records

        if self.mode == DataMode.OFFLINE_MOCK:
            records = self._mock_records(symbols)
        elif self.mode == DataMode.NETWORK:
            records = self._network_records(symbols, request.timeout)
        else:
            raise DataUnavailable(f"Unsupported data mode: {self.mode}")

        return records[:max_records]

    def _mock_records(self, symbols: list[str]) -> list[RawRecord]:
        now = _utc_now()
        records: list[RawRecord] = []
        # Deterministic synthetic session dates (Wed-Fri 2023-01-04..06).
        session_dates = [date(2023, 1, 4), date(2023, 1, 5), date(2023, 1, 6)]
        for symbol in symbols:
            base = 100.0 + hash(symbol) % 50
            for i, session_date in enumerate(session_dates):
                payload = {
                    "symbol": symbol,
                    "session_date": session_date,
                    "interval": "1d",
                    "open": round(base + i * 0.5, 2),
                    "high": round(base + i * 0.5 + 1.0, 2),
                    "low": round(base + i * 0.5 - 0.5, 2),
                    "close": round(base + i * 0.5 + 0.7, 2),
                    "volume": 1_000_000 + i * 100_000,
                    "currency": self.currency,
                    "exchange_mic": self.exchange_mic,
                }
                records.append(
                    RawRecord(
                        source_id="mock.market",
                        source_uri="mock://market/daily",
                        source_kind=SourceKind.MARKET,
                        provider_id=f"{symbol}|{session_date.isoformat()}",
                        published_time=None,
                        ingested_at=now,
                        available_at=now,
                        content_hash=content_hash(payload),
                        license_id="mock",
                        payload=payload,
                        quality_flags=[DAILY_SESSION_DATE_ONLY],
                    )
                )
        return records

    def _network_records(self, symbols: list[str], timeout: int) -> list[RawRecord]:
        try:
            import yfinance as yf
        except ImportError as exc:  # pragma: no cover
            raise DataUnavailable("yfinance not installed") from exc

        def download_symbol(sym: str) -> Any:
            return yf.download(
                sym,
                period="1mo",
                interval="1d",
                progress=False,
                auto_adjust=True,
                timeout=timeout,
            )

        now = _utc_now()
        records: list[RawRecord] = []
        for sym in symbols:
            try:
                df = with_retry(lambda sym=sym: download_symbol(sym))
                if df is None or df.empty:
                    continue
                df = df.reset_index()
                for _, row in df.iterrows():
                    ts = row.get("Date") or row.get("Datetime")
                    session_date: date
                    if hasattr(ts, "to_pydatetime"):
                        session_date = ts.to_pydatetime().date()
                    elif isinstance(ts, date):
                        session_date = ts
                    elif isinstance(ts, str):
                        session_date = date.fromisoformat(ts[:10])
                    else:
                        continue

                    payload = {
                        "symbol": sym,
                        "session_date": session_date,
                        "interval": "1d",
                        "open": float(row["Open"]),
                        "high": float(row["High"]),
                        "low": float(row["Low"]),
                        "close": float(row["Close"]),
                        "volume": float(row["Volume"]),
                        "currency": self.currency,
                        "exchange_mic": self.exchange_mic,
                    }
                    records.append(
                        RawRecord(
                            source_id="yfinance.daily",
                            source_uri=f"https://finance.yahoo.com/quote/{sym}/history",
                            source_kind=SourceKind.MARKET,
                            provider_id=f"{sym}|{session_date.isoformat()}",
                            published_time=None,
                            ingested_at=now,
                            available_at=now,
                            content_hash=content_hash(payload),
                            license_id="yfinance",
                            payload=payload,
                            quality_flags=[DAILY_SESSION_DATE_ONLY],
                        )
                    )
            except Exception as exc:  # noqa: BLE001
                raise DataUnavailable(f"yfinance download failed for {sym}") from exc
        return records


# ---------------------------------------------------------------------------
# RSS / document adapter
# ---------------------------------------------------------------------------


def _fetch_rss_items(feed_url: str, timeout: int) -> list[dict[str, Any]]:
    """Fetch and parse one RSS/Atom feed.

    Returns items with ``title``, ``link``, ``summary``, ``published`` (raw
    string), ``published_time`` (parsed UTC datetime or None) and
    ``published_quality_flags``.
    """
    req = urllib.request.Request(feed_url, headers={"User-Agent": "HermesBot/1.0"})

    def fetch() -> bytes:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()

    data = with_retry(fetch)
    root = ET.fromstring(data)

    items: list[dict[str, Any]] = []
    for item in root.iter("item") or root.iter("entry"):
        title = (item.findtext("title") or "").strip()
        link = item.findtext("link")
        if link is None:
            link_el = item.find("link")
            link = link_el.get("href") if link_el is not None else ""
        summary = (item.findtext("description") or item.findtext("summary") or "").strip()
        pub_raw = item.findtext("pubDate") or item.findtext("published") or ""
        pub_time, flags = parse_source_datetime(pub_raw)
        items.append(
            {
                "title": title,
                "link": link or "",
                "summary": summary,
                "published": pub_raw,
                "published_time": pub_time,
                "published_quality_flags": flags,
                "source": feed_url,
            }
        )
    return items


def _mock_rss_records(
    source_kind: SourceKind,
    items: list[dict[str, Any]],
    source_id: str,
) -> list[RawRecord]:
    now = _utc_now()
    records: list[RawRecord] = []
    for it in items:
        payload = {
            "title": it["title"],
            "summary": it["summary"],
            "language": "en",
            "document_type": source_kind.value,
        }
        published_raw = it.get("published", "")
        published_time, flags = parse_source_datetime(published_raw)
        if it.get("link"):
            provider_id = it["link"]
        else:
            provider_id = None
            flags.append(MISSING_PROVIDER_ID)
        records.append(
            RawRecord(
                source_id=source_id,
                source_uri=it.get("link") or f"mock://{source_id}/{len(records)}",
                source_kind=source_kind,
                provider_id=provider_id,
                published_time=published_time,
                ingested_at=now,
                available_at=now,
                content_hash=content_hash(payload),
                license_id="mock",
                payload=payload,
                quality_flags=flags,
                metadata={"published_raw": published_raw},
            )
        )
    return records


class RSSAdapter:
    """Adapter for RSS/Atom speech, report and alert feeds.

    ``OFFLINE_MOCK`` returns deterministic synthetic documents.  ``NETWORK``
    fetches configured feeds via ``urllib.request`` with bounded retry.
    """

    def __init__(
        self,
        mode: DataMode = DataMode.OFFLINE_MOCK,
        feeds: dict[str, str] | None = None,
        source_kind: SourceKind = SourceKind.ALERT,
        source_id: str = "mock.rss",
    ) -> None:
        self.mode = mode
        self.feeds = feeds or {}
        self.source_kind = source_kind
        self.source_id = source_id

    def fetch(self, request: AdapterRequest) -> list[RawRecord]:
        max_records = request.max_records
        if self.mode == DataMode.OFFLINE_MOCK:
            records = self._mock_records()
        elif self.mode == DataMode.NETWORK:
            records = self._network_records(request.timeout)
        else:
            raise DataUnavailable(f"Unsupported data mode: {self.mode}")
        return records[:max_records]

    def _mock_records(self) -> list[RawRecord]:
        # Synthetic fixture set; never exceeds the default budget.
        items: list[dict[str, Any]]
        if self.source_kind == SourceKind.SPEECH:
            items = [
                {
                    "title": "President Trump: remarks on infrastructure spending",
                    "summary": (
                        "We are investing heavily in domestic manufacturing and "
                        "grid infrastructure."
                    ),
                    "published": "Mon, 02 Jan 2023 14:00:00 GMT",
                    "link": "https://example.com/trump/remarks",
                },
                {
                    "title": "SEMI CEO earnings call: capacity constraints",
                    "summary": "Our semiconductor equipment order backlog is at record levels.",
                    "published": "2023-01-03T09:30:00+00:00",
                    "link": "https://example.com/ceo/earnings",
                },
            ]
        elif self.source_kind == SourceKind.REPORT:
            items = [
                {
                    "title": "NVDA quarterly results beat expectations",
                    "summary": "Revenue grew 65% YoY on AI chip demand.",
                    "published": "Tue, 03 Jan 2023 16:00:00 GMT",
                    "link": "https://example.com/nvda/q",
                },
                {
                    "title": "US government infrastructure spending bill",
                    "summary": (
                        "New federal budget allocates $50B to semiconductor and "
                        "grid infrastructure, targeting domestic manufacturing capacity."
                    ),
                    "published": "not-a-date",
                    "link": "https://example.com/gov/infra",
                },
            ]
        else:
            items = [
                {
                    "title": "Supply chain bottleneck intensifies",
                    "summary": (
                        "Actuator and power semiconductor shortages worsen as "
                        "demand outpaces capacity."
                    ),
                    "published": "Wed, 04 Jan 2023 08:00:00 GMT",
                    "link": "https://example.com/news/bottleneck",
                },
            ]
        return _mock_rss_records(self.source_kind, items, self.source_id)

    def _network_records(self, timeout: int) -> list[RawRecord]:
        now = _utc_now()
        records: list[RawRecord] = []
        for feed_name, feed_url in self.feeds.items():
            try:
                items = _fetch_rss_items(feed_url, timeout)
            except Exception as exc:  # noqa: BLE001
                raise DataUnavailable(f"RSS fetch failed for {feed_name}") from exc
            for it in items:
                payload = {
                    "title": it["title"],
                    "summary": it["summary"],
                    "language": "en",
                    "document_type": self.source_kind.value,
                    "feed": feed_name,
                }
                published_raw = it.get("published", "")
                published_time = it.get("published_time")
                flags = list(it.get("published_quality_flags", []))
                provider_id = it.get("link") or None
                if not provider_id:
                    flags.append(MISSING_PROVIDER_ID)
                records.append(
                    RawRecord(
                        source_id=f"{self.source_id}.{feed_name}",
                        source_uri=it.get("link") or feed_url,
                        source_kind=self.source_kind,
                        provider_id=provider_id,
                        published_time=published_time,
                        ingested_at=now,
                        available_at=now,
                        content_hash=content_hash(payload),
                        license_id="rss",
                        payload=payload,
                        quality_flags=flags,
                        metadata={"published_raw": published_raw, "feed": feed_name},
                    )
                )
        return records
