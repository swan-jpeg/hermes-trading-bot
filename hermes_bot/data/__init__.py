"""LAYER 1-2: data collection + storage. See obsidian [[02-Data-Laag]]."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CollectorConfig:
    """Per-route collector configuratie."""

    name: str
    enabled: bool = True
    interval_sec: int = 3600
    retries: int = 3


class BaseCollector:
    """Interface for every data collector (speech/reports/alerts/market)."""

    name = "base"

    def __init__(self, cfg: CollectorConfig | None = None) -> None:
        self.cfg = cfg or CollectorConfig(name=self.name)

    def collect(self) -> list[dict]:
        """Fetch raw records. Must be idempotent + structured."""
        raise NotImplementedError

    def health(self) -> dict[str, object]:
        return {"name": self.name, "enabled": self.cfg.enabled}
