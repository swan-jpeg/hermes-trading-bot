"""
Tests for the GDELT replay module.
"""

import json
import unittest
from unittest.mock import MagicMock, patch

from hermes_bot.replay.gdelt import fetch_events


class TestGDELTFetchEvents(unittest.TestCase):
    
    def test_fetch_events_empty_response(self):
        """Test fetch_events with empty response"""
        with patch('hermes_bot.replay.gdelt.urlopen') as mock_urlopen:
            # Mock empty response
            mock_response = MagicMock()
            mock_response.read.return_value = b'{"articles": []}'
            mock_urlopen.return_value.__enter__.return_value = mock_response
            
            result = fetch_events("test query", "2023-01-01", "2023-01-02")
            
            self.assertEqual(result, [])
    
    def test_fetch_events_with_data(self):
        """Test fetch_events with sample data"""
        sample_data = {
            "articles": [
                {
                    "date": "2023-01-01",
                    "title": "Sample event title",
                    "url": "https://example.com/article",
                    "domain": "example.com",
                    "source": "news"
                }
            ]
        }
        
        with patch('hermes_bot.replay.gdelt.urlopen') as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = json.dumps(sample_data).encode()
            mock_urlopen.return_value.__enter__.return_value = mock_response
            
            result = fetch_events("test query", "2023-01-01", "2023-01-02")
            
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["date"], "2023-01-01")
            self.assertEqual(result[0]["title"], "Sample event title")
            self.assertEqual(result[0]["url"], "https://example.com/article")
            self.assertEqual(result[0]["domain"], "example.com")
            self.assertEqual(result[0]["source"], "news")
    
    def test_fetch_events_network_error(self):
        """Test fetch_events with network error"""
        with patch('hermes_bot.replay.gdelt.urlopen') as mock_urlopen:
            mock_urlopen.side_effect = Exception("Network error")
            
            result = fetch_events("test query", "2023-01-01", "2023-01-02")
            
            # Should return empty list on network error (offline-safe)
            self.assertEqual(result, [])
    
    def test_fetch_events_json_error(self):
        """Test fetch_events with JSON decode error"""
        with patch('hermes_bot.replay.gdelt.urlopen') as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = b'invalid json'
            mock_urlopen.return_value.__enter__.return_value = mock_response
            
            result = fetch_events("test query", "2023-01-01", "2023-01-02")
            
            # Should return empty list on JSON error
            self.assertEqual(result, [])


if __name__ == '__main__':
    unittest.main()