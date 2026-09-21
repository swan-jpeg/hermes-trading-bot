"""LAAG 1-2: datacollectie + opslag. Zie obsidian [[02-Data-Laag]]."""
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
    """Interface voor elke data-collector (speech/reports/alerts/market)."""

    name = "base"

    def __init__(self, cfg: CollectorConfig | None = None) -> None:
        self.cfg = cfg or CollectorConfig(name=self.name)

    def collect(self) -> list[dict]:
        """Haal ruwe records op. Must be idempotent + gestructureerd."""
        raise NotImplementedError

    def health(self) -> dict[str, object]:
        return {"name": self.name, "enabled": self.cfg.enabled}
