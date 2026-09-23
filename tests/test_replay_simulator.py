"""
Tests for the Replay Simulator module.
"""

import logging
import unittest
from unittest.mock import patch

from hermes_bot.replay.simulator import HistoricalEvent, ReplaySimulator

# Setup logging to suppress warnings during tests
logging.basicConfig(level=logging.WARNING)



class TestReplaySimulator(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a minimal simulator for testing
        self.simulator = ReplaySimulator(
            sectors=["government", "tech"],
            start_date="2023-01-01",
            end_date="2023-01-05"
        )
    
    def test_init(self):
        """Test initialization of ReplaySimulator."""
        self.assertIsNotNone(self.simulator.impact_agent)
        self.assertEqual(self.simulator.sectors, ["government", "tech"])
        self.assertEqual(self.simulator.start_date, "2023-01-01")
        self.assertEqual(self.simulator.end_date, "2023-01-05")
        self.assertEqual(self.simulator.events_by_date, {})
    
    def test_historical_event_dataclass(self):
        """Test HistoricalEvent dataclass."""
        event = HistoricalEvent(
            date="2023-01-01",
            title="Test event",
            url="https://example.com/test",
            domain="example.com",
            source="news"
        )
        
        self.assertEqual(event.date, "2023-01-01")
        self.assertEqual(event.title, "Test event")
        self.assertEqual(event.url, "https://example.com/test")
        self.assertEqual(event.domain, "example.com")
        self.assertEqual(event.source, "news")
    
    @patch('hermes_bot.replay.simulator.fetch_events')
    def test_fetch_historical_events(self, mock_fetch):
        """Test fetching historical events."""
        # Mock the fetch_events function to return sample data
        # We expect 2 events because we have 2 sectors in setUp
        mock_fetch.return_value = [
            {
                "date": "2023-01-01",
                "title": "Test event 1",
                "url": "https://example.com/test1",
                "domain": "example.com",
                "source": "news"
            }
        ]
        
        # Call the method
        self.simulator.fetch_historical_events()
        
        # Check that events were stored correctly
        self.assertIn("2023-01-01", self.simulator.events_by_date)
        # We expect 2 events for this date (because we have 2 sectors in the setup)
        self.assertEqual(len(self.simulator.events_by_date["2023-01-01"]), 2)
    
    def test_load_prices_for_date_range(self):
        """Test loading price data (real yfinance, or simulated fallback)."""
        # Call the method
        self.simulator.load_prices_for_date_range("SPY")

        # Check that prices were loaded (real yfinance gives a few points
        # for a 5-day range; the simulated fallback gives 252).
        self.assertIn("SPY", self.simulator.prices)
        self.assertGreater(len(self.simulator.prices["SPY"]), 0)
    
    def test_process_day_no_events(self):
        """Test processing a day with no events."""
        # Ensure no events exist for this date
        contexts = self.simulator.process_day("2023-01-01", "SPY")
        self.assertEqual(contexts, [])


if __name__ == '__main__':
    unittest.main()