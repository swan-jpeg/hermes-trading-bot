"""LAAG 7: Alpaca broker integratie (paper)."""
from __future__ import annotations

from typing import TYPE_CHECKING

from hermes_bot.execution import BaseBroker, Order

if TYPE_CHECKING:
    pass


class AlpacaBroker(BaseBroker):
    """Alpaca paper trading broker implementatie."""

    def __init__(self, api_key: str | None = None, secret_key: str | None = None) -> None:
        super().__init__()
        
        # For T7 this is an offline-safe implementation
        # In a real implementation we would connect the Alpaca API here
        self.api_key = api_key
        self.secret_key = secret_key
        self._connected = False

    def submit(self, order: Order) -> Order:
        """Place an order via the Alpaca API.

        Fail-closed: als broker niet beschikbaar is, wordt de order geweigerd.
        """
        # For T7: offline-safe implementation
        # In a real implementation we would call the Alpaca API here
        
        if not self.health():
            # Fail-closed: if the broker is unavailable, return a rejected order
            order.status = "rejected"
            return order
        
        # For now a dummy implementation
        # In a real implementation we would here:
        # 1. Connect to the Alpaca API
        # 2. Place the order
        # 3. Process the response
        
        order.status = "submitted"
        return order

    def get_positions(self) -> dict[str, float]:
        """Fetch current positions via the Alpaca API."""
        # For T7: offline-safe implementation
        # In a real implementation we would call the Alpaca API here
        
        if not self.health():
            return {}
        
        # For now a dummy implementation
        # In a real implementation we would here:
        # 1. Connect to the Alpaca API
        # 2. Fetch the positions
        # 3. Process the results
        
        return {"AAPL": 10.0, "MSFT": 5.0}  # Dummy data

    def health(self) -> bool:
        """Check whether the broker is available."""
        # For T7: offline-safe implementation
        # In a real implementation we would here:
        # 1. Connect to the Alpaca API
        # 2. Status controleren
        # 3. Resultaat retourneren
        
        # In T7 this is a simulation of an offline-safe check
        # In a real implementation this would be a real API call
        try:
            # Simulate that we can reach the API
            # Always available for now
            self._connected = True
            return True
        except Exception:
            self._connected = False
            return False