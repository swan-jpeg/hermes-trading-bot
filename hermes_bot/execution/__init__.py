"""LAYER 7: execution — output action to broker (paper/live). [[10]]

Principe: decision != executie. Deze laag is de enige plek die orders
naar de broker stuurt. Fail-closed: geen antwoord -> geen order.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Order:
    asset: str
    side: str  # buy | sell
    qty: float
    order_type: str = "market"  # market | limit
    limit_price: float | None = None
    status: str = "new"


class BaseBroker:
    """Interface over brokers. Implement per broker (alpaca/ibkr)."""

    name = "base"

    def submit(self, order: Order) -> Order:
        """Send; set status to filled/partially_filled/rejected."""
        raise NotImplementedError

    def get_positions(self) -> dict[str, float]:
        raise NotImplementedError

    def health(self) -> bool:
        raise NotImplementedError


class PaperBroker(BaseBroker):
    """Simulates fills without real money. Mandatory first step."""

    name = "paper"

    def __init__(self) -> None:
        self._positions: dict[str, float] = {}
        self._fills: list[dict] = []

    def submit(self, order: Order) -> Order:
        # Paper: always filled at market price (no slippage in v1).
        order.status = "filled"
        self._positions[order.asset] = self._positions.get(order.asset, 0.0) + (
            order.qty if order.side == "buy" else -order.qty
        )
        self._fills.append({"asset": order.asset, "side": order.side, "qty": order.qty})
        return order

    def get_positions(self) -> dict[str, float]:
        return dict(self._positions)

    def health(self) -> bool:
        return True


class ExecutionEngine:
    def __init__(self, broker: BaseBroker, fail_closed: bool = True) -> None:
        self.broker = broker
        self.fail_closed = fail_closed
        self._log: list[dict] = []

    def execute(self, orders: list[Order]) -> list[Order]:
        if self.fail_closed and not self.broker.health():
            raise RuntimeError("Broker onbereikbaar -> fail-closed, geen orders uitgevoerd")
        out: list[Order] = []
        for o in orders:
            filled = self.broker.submit(o)
            self._log.append(
                {"asset": o.asset, "side": o.side, "qty": o.qty, "status": filled.status}
            )
            out.append(filled)
        return out

    def fills(self) -> list[dict]:
        return self._log
