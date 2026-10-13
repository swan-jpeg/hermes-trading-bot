"""LAAG 7: Alpaca broker adapter — DISABLED STUB in de MVP."""
from __future__ import annotations

from hermes_bot.execution import BaseBroker, Order


class AlpacaBroker(BaseBroker):
    """Alpaca broker adapter — bewust uitgeschakeld in de MVP.

    Er worden geen Alpaca API-calls gedaan, geen credentials gelezen en geen
    netwerkconnectie opgebouwd. ``health()`` retourneert altijd ``False``; dit
    is een zichtbare stub en mag nooit worden geïnterpreteerd als een werkende
    paper- of live-verbinding.
    """

    name = "alpaca_disabled"

    def __init__(self, api_key: str | None = None, secret_key: str | None = None) -> None:
        super().__init__()
        self.api_key = api_key
        self.secret_key = secret_key
        self._disabled_reason = "Alpaca broker is disabled in the MVP (mock-only execution)"

    def submit(self, order: Order) -> Order:
        """Fail-closed: reject every order because the adapter is disabled."""
        order.status = "rejected"
        return order

    def get_positions(self) -> dict[str, float]:
        """No real positions: the adapter is not connected."""
        return {}

    def health(self) -> bool:
        """Always False.

        ``True`` zou impliceren dat er een echte connectie is; de MVP draait
        uitsluitend op de in-memory MockBroker.
        """
        return False

    @property
    def disabled_reason(self) -> str:
        return self._disabled_reason
