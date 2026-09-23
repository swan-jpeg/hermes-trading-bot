"""
Simple tests for the Replay Simulator module.
"""

import logging
import unittest

from hermes_bot.replay.simulator import HistoricalEvent, ReplaySimulator

# Setup logging to suppress warnings during tests
logging.basicConfig(level=logging.WARNING)



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
        """Test loading price data (real yfinance, or synthetic fallback)."""
        simulator = ReplaySimulator()
        simulator.load_prices_for_date_range("SPY")
        # Prices should be loaded (real yfinance, or synthetic fallback).
        self.assertIn("SPY", simulator.prices)
        self.assertGreater(len(simulator.prices["SPY"]), 0)


if __name__ == '__main__':
    unittest.main()
