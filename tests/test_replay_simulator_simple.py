"""
Simple tests for the Replay Simulator module.
"""

import unittest
from unittest.mock import patch, MagicMock
import logging

# Setup logging to suppress warnings during tests
logging.basicConfig(level=logging.WARNING)

from hermes_bot.replay.simulator import ReplaySimulator, HistoricalEvent


class TestReplaySimulator(unittest.TestCase):
    
    def test_init(self):
        """Test initialization of ReplaySimulator."""
        simulator = ReplaySimulator(
            sectors=["government", "tech"],
            start_date="2023-01-01",
            end_date="2023-01-05"
        )
        self.assertIsNotNone(simulator.impact_agent)
        self.assertEqual(simulator.sectors, ["government", "tech"])
        self.assertEqual(simulator.start_date, "2023-01-01")
        self.assertEqual(simulator.end_date, "2023-01-05")
        self.assertEqual(simulator.events_by_date, {})
    
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
    
    def test_load_prices_for_date_range(self):
        """Test loading price data (simulated)."""
        simulator = ReplaySimulator()
        simulator.load_prices_for_date_range("SPY")
        
        # Check that prices were loaded
        self.assertIn("SPY", simulator.prices)
        self.assertEqual(len(simulator.prices["SPY"]), 252)


if __name__ == '__main__':
    unittest.main()