"""CLI entry point for the bot."""
from __future__ import annotations

from typing import Any

import fire

from hermes_bot.config import load_config


class HermesBotCLI:
    """Hermes Trading Bot CLI."""

    def __init__(self) -> None:
        self.config: dict[str, Any] = load_config()

    def hello(self) -> str:
        return "Hermes trading-bot ready (skeleton)."

    def universe(self) -> list[str]:
        return self.config.get("universe", {}).get("stocks", [])

    def web_status(self) -> dict:
        """Show which webscraping routes are enabled (from config.yaml)."""
        ws = self.config.get("webscraping", {})
        routes = ws.get("routes", {})
        return {
            "master_enabled": ws.get("enabled", False),
            "routes": {
                k: {
                    "enabled": v.get("enabled", False),
                    "interval_sec": v.get("interval_sec", 0),
                }
                for k, v in routes.items()
            },
        }

    def web_scrape(self, use_demo: bool = False) -> dict:
        """Run the webscraping->fusion chain (offline demo or real feeds)."""
        from hermes_bot.data.scraper import WebScraper, scrape_and_fuse
        ws = WebScraper.from_config_file()
        if use_demo or not ws.cfg.get("enabled", False):
            ws = WebScraper({"enabled": False})
        return scrape_and_fuse(ws)


def main() -> None:
    fire.Fire(HermesBotCLI)


if __name__ == "__main__":
    main()
