"""
Replay Simulator for historical event training.

Simulates live event processing by replaying historical events day-by-day,
processing them through the impact agent, fusion, and RL pipeline.
"""

import logging
import os
import pathlib
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from hermes_bot.impact import LLMImpactAgent
from hermes_bot.portfolio import PortfolioState

# Import necessary modules
from hermes_bot.replay.gdelt import fetch_events
from hermes_bot.rl.context import AssetContext, build_asset_context

logger = logging.getLogger(__name__)


@dataclass
class HistoricalEvent:
    """Represents a historical event fetched from GDELT."""
    date: str
    title: str
    url: str
    domain: str
    source: str


class ReplaySimulator:
    """Simulates live event processing using historical GDELT events."""
    
    def __init__(
        self,
        impact_agent: LLMImpactAgent | None = None,
        sectors: list[str] | None = None,
        start_date: str = "2020-01-01",
        end_date: str = "2023-12-31",
        data_dir: str | None = None,
    ):
        """
        Initialize the replay simulator.

        Args:
            impact_agent: The impact agent to use for processing events.
                Default: LLMImpactAgent with the best FREE Nous model
                (stepfun/step-3.7-flash:free) — NOT a paid model.
            sectors: List of sectors to query for historical events.
            start_date: Start date for historical events (YYYY-MM-DD).
            end_date: End date for historical events (YYYY-MM-DD).
            data_dir: Where to store large replay data (default: the
                REPLAY_DATA_DIR env-var, which points to the external SSD).
        """
        if impact_agent is None:
            impact_agent = LLMImpactAgent(
                model="stepfun/step-3.7-flash:free",
                base_url="https://inference-api.nousresearch.com/v1",
            )
        self.impact_agent = impact_agent
        self.sectors = sectors or [
            "government", "macro", "tech", "consumer", "healthcare",
            "finance", "semiconductor", "actuators", "power", "battery"
        ]
        self.start_date = start_date
        self.end_date = end_date
        self.events_by_date: dict[str, list[HistoricalEvent]] = defaultdict(list)
        self.prices: dict[str, list[float]] = {}
        self.data_dir = pathlib.Path(data_dir or os.environ.get(
            "REPLAY_DATA_DIR", "/mnt/ssd/trading-bot-replay"))
        
    def fetch_historical_events(self) -> None:
        """
        Fetch historical events for all sectors within the date range.
        """
        logger.info(f"Fetching historical events from {self.start_date} to {self.end_date}")
        
        # Clear existing events
        self.events_by_date.clear()
        
        # Fetch events for each sector (with a pause to respect GDELT's
        # rate limit — 429s happen when hammering the API).
        for i, sector in enumerate(self.sectors):
            if i > 0:
                time.sleep(2.0)  # avoid 429 rate-limit
            logger.info(f"Fetching events for sector: {sector}")
            events = fetch_events(
                query=sector,
                start=self.start_date,
                end=self.end_date,
                maxrecords=50  # Limit to avoid too many requests
            )
            
            # Convert to HistoricalEvent objects and group by date
            for event_data in events:
                event = HistoricalEvent(
                    date=event_data["date"],
                    title=event_data["title"],
                    url=event_data["url"],
                    domain=event_data["domain"],
                    source=event_data["source"]
                )
                self.events_by_date[event.date].append(event)
                
        total = sum(len(ev) for ev in self.events_by_date.values())
        logger.info(f"Fetched {total} events across {len(self.events_by_date)} days")
    
    def load_prices_for_date_range(self, ticker: str) -> None:
        """
        Load price data for the date range used in simulation.
        
        Args:
            ticker: The stock ticker to load prices for.
        """
        logger.info(f"Loading prices for {ticker}")
        try:
            import yfinance as yf
            
            # Download price data for the date range
            df = yf.download(ticker, start=self.start_date, end=self.end_date,
                             progress=False, auto_adjust=True)
            prices = df["Close"].values
            
            # Remove any NaN values
            prices = prices[~np.isnan(prices)]
            
            # Convert to list of floats
            self.prices[ticker] = prices.tolist()
            
            logger.info(f"Fetched {len(self.prices[ticker])} price points for {ticker}")
        except Exception as e:
            # Fallback to simulated prices if yfinance fails
            logger.warning(f"Failed to fetch real prices for {ticker}: {e}. "
                           "Using simulated prices.")
            # Simulate some prices as a fallback
            self.prices[ticker] = [100.0] * 252  # Default to constant price
            logger.info(f"Simulated {len(self.prices[ticker])} price points for {ticker}")
    
    def process_day(
        self, 
        date: str, 
        ticker: str = "SPY",
        portfolio: PortfolioState | None = None,
        seed: int = 42
    ) -> list[AssetContext]:
        """
        Process all events for a given day through the full pipeline.
        
        Args:
            date: Date to process (YYYY-MM-DD).
            ticker: Stock ticker to simulate (default: SPY).
            portfolio: Portfolio state to use (default: new portfolio).
            seed: Random seed for reproducibility.
            
        Returns:
            List of AssetContext objects representing the state after processing
            all events for the day.
        """
        # Get events for this date
        events_for_day = self.events_by_date.get(date, [])
        if not events_for_day:
            logger.debug(f"No events for date {date}")
            return []
        
        logger.info(f"Processing {len(events_for_day)} events for {date}")
        
        # Initialize portfolio if not provided
        if portfolio is None:
            portfolio = PortfolioState(cash=100_000.0)
        
        # Generate synthetic fusion signals for the day (for the RL environment)
        # This represents the "market context" for the day
        # For now, we'll simulate what build_fusion_signals would return
        if ticker not in self.prices:
            self.load_prices_for_date_range(ticker)
        
        # In a real implementation, the fusion model output would be used here.
        # For now the AssetContexts carry the market interpretation directly.
        
        # Process each event through the impact agent
        all_asset_contexts = []
        
        # Process events sequentially for this day
        for i, event in enumerate(events_for_day):
            logger.debug(f"Processing event {i+1}/{len(events_for_day)}: {event.title[:100]}...")
            
            # Use the impact agent to determine which entities are affected
            impact_result = self.impact_agent.analyze(
                event=event.title,
                source=event.source,
                entity_id=""  # No specific entity ID for this replay
            )
            
            # Convert impact result to fusion-style inputs
            fusion_inputs = self.impact_agent.to_fusion_inputs(impact_result, sentiment=0.0)
            
            # For each impacted entity, build an AssetContext
            for fusion_input in fusion_inputs:
                entity = fusion_input["entity_id"]
                asset_class = fusion_input["asset_class"]
                
                # Build the asset context using the impact, fusion, and other data
                ctx = build_asset_context(
                    impact={
                        "direction": impact_result.direction,
                        "magnitude": impact_result.magnitude,
                        "confidence": impact_result.confidence,
                        "probability": impact_result.probability,
                        "duration": impact_result.duration,
                        "directness": impact_result.directness,
                        "novelty": impact_result.novelty,
                        "surprise": impact_result.surprise,
                    },
                    fusion={
                        "emotie": {
                            "sentiment": fusion_input["sentiment"]
                        },
                        "zekerheid": fusion_input["confidence"],
                        "market_expectation": 0.5,
                        "expectation_surprise": 0.0,
                        "narrative_strength": 0.0,
                        "market_attention": 0.0,
                        "consensus_strength": 0.5,
                        "contrarian_strength": 0.0,
                    },
                    bottleneck={},  # Empty for now, could be extended
                    source={
                        "quality": 0.7,
                        "reliability": 0.7,
                        "confidence": impact_result.confidence,
                        "completeness": 0.6,
                        "cross_source_confirmation": 0.5,
                        "freshness": 0.8,
                        "novelty": 0.3,
                    },
                    entity=entity,
                    asset_class=asset_class,
                )
                
                all_asset_contexts.append(ctx)
        
        return all_asset_contexts
    
    def save_data(self, ticker: str = "SPY") -> None:
        """Save the fetched events + prices to the data dir (external SSD).

        Dit is de export die de gebruiker naar zijn PC downloadt om daar te
        trainen. Alleen code + een klein voorbeeld gaat naar GitHub; de grote
        data blijft op de externe SSD.
        """
        import json

        self.data_dir.mkdir(parents=True, exist_ok=True)
        events_out = []
        for _date, evs in self.events_by_date.items():
            for ev in evs:
                events_out.append({
                    "date": ev.date, "title": ev.title, "url": ev.url,
                    "domain": ev.domain, "source": ev.source,
                })
        (self.data_dir / "events.json").write_text(
            json.dumps(events_out, indent=2))
        (self.data_dir / "prices.json").write_text(
            json.dumps(self.prices, indent=2))
        logger.info(f"Data opgeslagen in {self.data_dir} "
                    f"({len(events_out)} events)")

    def load_data(self, ticker: str = "SPY") -> bool:
        """Load previously saved events + prices from the data dir.

        Returns True if data was loaded (so the user can train offline on the
        PC without re-fetching GDELT).
        """
        import json

        events_file = self.data_dir / "events.json"
        prices_file = self.data_dir / "prices.json"
        if not events_file.exists() or not prices_file.exists():
            return False
        self.events_by_date.clear()
        for ev in json.loads(events_file.read_text()):
            self.events_by_date[ev["date"]].append(HistoricalEvent(**ev))
        self.prices = json.loads(prices_file.read_text())
        logger.info(f"Data geladen uit {self.data_dir} "
                    f"({sum(len(v) for v in self.events_by_date.values())} events)")
        return True

    def simulate_replay(
        self, 
        ticker: str = "SPY",
        seed: int = 42
    ) -> dict[str, list[AssetContext]]:
        """
        Simulate the complete replay from start to end date.
        
        Args:
            ticker: Stock ticker to simulate (default: SPY).
            seed: Random seed for reproducibility.
            
        Returns:
            Dictionary mapping dates to lists of AssetContext objects.
        """
        logger.info(f"Starting replay simulation for {ticker} from "
                     f"{self.start_date} to {self.end_date}")
        
        # Process all days in the date range
        all_contexts = {}
        
        # Convert date strings to datetime objects
        start_dt = datetime.strptime(self.start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(self.end_date, "%Y-%m-%d")
        
        # Iterate through each day in the range
        current_date = start_dt
        while current_date <= end_dt:
            date_str = current_date.strftime("%Y-%m-%d")
            
            # Process the day
            contexts = self.process_day(date_str, ticker, seed=seed)
            all_contexts[date_str] = contexts
            
            # Move to next day
            current_date += timedelta(days=1)
        
        logger.info(f"Replay simulation completed. Generated contexts "
                     f"for {len(all_contexts)} days.")
        return all_contexts


# Example usage
if __name__ == "__main__":
    import sys
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Create simulator
        simulator = ReplaySimulator(
            sectors=["government", "macro", "tech"],
            start_date="2023-01-01",
            end_date="2023-01-05"
        )
        
        # Fetch events
        simulator.fetch_historical_events()
        
        # Simulate replay
        contexts = simulator.simulate_replay(ticker="SPY", seed=42)
        
        # Print results
        print(f"Generated contexts for {len(contexts)} days:")
        for date, ctx_list in list(contexts.items())[:3]:  # Show first 3 days
            print(f"  {date}: {len(ctx_list)} contexts")
            for i, ctx in enumerate(ctx_list[:2]):  # Show first 2 contexts per day
                print(f"    Context {i+1}: entity={ctx.entity}, "
                      f"impact_dir={ctx.impact_direction:.2f}")
        
        print("Replay simulation completed successfully!")
        
    except Exception as e:
        print(f"Error in replay simulation: {e}", file=sys.stderr)
        sys.exit(1)