"""
GDELT 2.0 Document API downloader.

Downloads historical events from GDELT 2.0 using the Document API.
Free, open, no API key required.
"""

import json
import logging
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


def _to_gdelt_dt(d: str) -> str:
    """Convert YYYY-MM-DD (or YYYYMMDD) to YYYYMMDD000000 for the GDELT API."""
    digits = "".join(ch for ch in d if ch.isdigit())
    if len(digits) >= 14:
        return digits[:14]
    if len(digits) >= 8:
        return digits[:8] + "000000"
    return digits + "000000"


def fetch_events(
    query: str,
    start: str,
    end: str,
    maxrecords: int = 250,
) -> list[dict]:
    """
    Fetch historical events from GDELT 2.0 Document API.

    Args:
        query: Search query (e.g. "semiconductor", "inflation", "Trump")
        start: Start date in YYYY-MM-DD format
        end: End date in YYYY-MM-DD format
        maxrecords: Maximum number of records to return (default 250)

    Returns:
        List of event dictionaries with keys: date, title, url, domain, source

    Example:
        >>> events = fetch_events("government spending", "2023-01-01", "2023-01-02")
        >>> print(events[0])
        {'date': '2023-01-01', 'title': 'Government announces new spending...',
         'url': 'https://example.com/article', 'domain': 'example.com',
         'source': 'news'}
    """
    base_url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "maxrecords": maxrecords,
        "startdatetime": _to_gdelt_dt(start),
        "enddatetime": _to_gdelt_dt(end),
    }
    query_string = urlencode(params)
    url = f"{base_url}?{query_string}"

    # Retry with backoff (GDELT is rate-limited; 429/SSL-timeouts happen).
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            request = Request(url)
            request.add_header("User-Agent", "TradingBot/1.0")
            with urlopen(request, timeout=30) as response:
                data = response.read()
            parsed_data = json.loads(data)
            break
        except HTTPError as e:
            last_err = e
            if e.code == 429:  # rate-limited: wait and retry
                time.sleep(2 * (attempt + 1))
                continue
            logger.warning(f"HTTP error fetching GDELT events: {e}")
            return []
        except (URLError, TimeoutError, OSError) as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error fetching GDELT events: {e}")
            return []
    else:
        logger.warning(f"Network error fetching GDELT events: {last_err}")
        return []

    # Transform the data to our standard format.
    events = []
    for item in parsed_data.get("articles", []):
        events.append({
            "date": item.get("date", ""),
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "domain": item.get("domain", ""),
            "source": item.get("source", ""),
        })
    logger.info(f"Fetched {len(events)} events for query '{query}' "
                f"from {start} to {end}")
    return events


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    events = fetch_events("government spending", "2023-01-01", "2023-01-02", 10)
    print(f"Retrieved {len(events)} events:")
    for event in events[:3]:
        print(f"  {event['date']}: {event['title'][:100]}...")
