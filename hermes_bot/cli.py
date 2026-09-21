"""CLI-entry voor de bot."""
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


def main() -> None:
    fire.Fire(HermesBotCLI)


if __name__ == "__main__":
    main()
