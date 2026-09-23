"""
GDELT 2.0 Document API downloader.

Downloads historical events from GDELT 2.0 using the Document API.
Free, open, no API key required.
"""

import json
import logging
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


def fetch_events(
    query: str,
    start: str,
    end: str,
    maxrecords: int = 250
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
    # Convert YYYY-MM-DD to YYYYMMDD000000 format for GDELT API
    def convert_date(date_str):
        # Assuming date_str is in YYYY-MM-DD format
        parts = date_str.split('-')
        return f"{parts[0]}{parts[1]}{parts[2]}000000"
    
    # Construct the API URL
    base_url = "https://api.gdeltproject.org/api/v2/doc/doc"
    
    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "maxrecords": maxrecords,
        "startdatetime": convert_date(start),
        "enddatetime": convert_date(end)
    }
    
    # Encode parameters
    query_string = urlencode(params)
    url = f"{base_url}?{query_string}"
    
    try:
        # Make the request with a timeout
        request = Request(url)
        request.add_header('User-Agent', 'TradingBot/1.0')
        
        with urlopen(request, timeout=10) as response:
            data = response.read()
            
        # Parse JSON response
        parsed_data = json.loads(data)
        
        # Transform the data to our standard format
        events = []
        for item in parsed_data.get("articles", []):
            event = {
                "date": item.get("date", ""),
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "domain": item.get("domain", ""),
                "source": item.get("source", "")
            }
            events.append(event)
            
        logger.info(f"Fetched {len(events)} events for query '{query}' from {start} to {end}")
        return events
        
    except URLError as e:
        logger.warning(f"Network error fetching GDELT events: {e}")
        # Return empty list for offline-safe behavior
        return []
    except json.JSONDecodeError as e:
        logger.warning(f"JSON decode error: {e}")
        return []
    except Exception as e:
        logger.warning(f"Unexpected error fetching GDELT events: {e}")
        return []


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Test with a simple query
    events = fetch_events("government spending", "2023-01-01", "2023-01-02", 10)
    print(f"Retrieved {len(events)} events:")
    for event in events[:3]:  # Show first 3
        print(f"  {event['date']}: {event['title'][:100]}...")