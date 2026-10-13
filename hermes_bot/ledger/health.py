"""Health monitoring: HEALTHY | DEGRADED | PAUSED."""
from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any

from hermes_bot.ledger.repository import HealthRepository
from hermes_bot.ledger.schemas import HealthState, HealthStatus


class HealthMonitor:
    """Evaluate pipeline health and persist the current state.

    The monitor never emits trading side effects; callers must check the state
    and pause execution themselves.
    """

    def __init__(self, repo: HealthRepository) -> None:
        self.repo = repo

    def get(self) -> HealthState:
        state = self.repo.get()
        if state is None:
            state = HealthState(
                state=HealthStatus.HEALTHY,
                reason="initial",
                updated_at=datetime.now(UTC),
            )
        return state

    def set(self, state: HealthStatus, reason: str) -> HealthState:
        value = HealthState(state=state, reason=reason, updated_at=datetime.now(UTC))
        self.repo.set(value)
        return value

    def evaluate_prices(
        self,
        prices: dict[str, float],
        as_of: datetime,
        max_age_seconds: float,
        price_available_at: datetime | None = None,
    ) -> HealthState:
        """Stale or missing prices move health to PAUSED."""
        if not prices:
            return self.set(HealthStatus.PAUSED, "no prices available")
        if any(not math.isfinite(p) or p <= 0 for p in prices.values()):
            return self.set(HealthStatus.PAUSED, "invalid or missing price")
        if price_available_at is not None:
            age = (as_of - price_available_at).total_seconds()
            if age > max_age_seconds:
                return self.set(HealthStatus.PAUSED, f"stale price: age={age:.0f}s")
        return self.get()

    def evaluate_broker_state(self, has_unknown_orders: bool) -> HealthState:
        """Unknown broker orders move health to PAUSED until reconciliation."""
        if has_unknown_orders:
            return self.set(
                HealthStatus.PAUSED,
                "unknown broker orders require reconciliation",
            )
        return self.get()

    def evaluate_storage_write(self, success: bool, detail: str = "") -> HealthState:
        """Failed storage writes move health to PAUSED (fail closed)."""
        if not success:
            reason = "storage write failed"
            if detail:
                reason += f": {detail}"
            return self.set(HealthStatus.PAUSED, reason)
        return self.get()

    def evaluate_risk_config(self, risk_cfg: dict[str, Any] | None) -> HealthState:
        """Invalid or missing risk configuration moves health to PAUSED."""
        if risk_cfg is None:
            return self.set(HealthStatus.PAUSED, "risk config missing")
        required = {"max_gross_exposure", "max_sector_weight", "max_drawdown", "min_cash"}
        missing = required - set(risk_cfg.keys())
        if missing:
            return self.set(HealthStatus.PAUSED, f"risk config missing keys: {sorted(missing)}")
        for key, value in risk_cfg.items():
            if key in required:
                valid = isinstance(value, (int, float)) and math.isfinite(value)
                if not valid:
                    return self.set(
                        HealthStatus.PAUSED, f"risk config key '{key}' invalid: {value}"
                    )
        return self.get()

    def mark_healthy(self, reason: str = "ok") -> HealthState:
        return self.set(HealthStatus.HEALTHY, reason)

    def mark_degraded(self, reason: str) -> HealthState:
        return self.set(HealthStatus.DEGRADED, reason)
