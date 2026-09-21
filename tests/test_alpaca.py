"""Tests voor Alpaca broker integratie (T7)."""
from __future__ import annotations

from hermes_bot.execution import Order
from hermes_bot.execution.brokers.alpaca import AlpacaBroker


def test_alpaca_broker_creation() -> None:
    """Test dat AlpacaBroker correct wordt aangemaakt."""
    broker = AlpacaBroker()
    
    # Controleer dat het een BaseBroker is
    assert hasattr(broker, 'submit')
    assert hasattr(broker, 'get_positions')
    assert hasattr(broker, 'health')


def test_alpaca_broker_health() -> None:
    """Test dat health() werkt."""
    broker = AlpacaBroker()
    
    # Test dat de health methode werkt
    health_status = broker.health()
    assert isinstance(health_status, bool)


def test_alpaca_broker_submit() -> None:
    """Test dat submit() werkt."""
    broker = AlpacaBroker()
    
    # Maak een dummy order
    order = Order(asset="AAPL", side="buy", qty=10.0)
    
    # Test submit
    result = broker.submit(order)
    
    # Controleer dat het resultaat correct is
    assert hasattr(result, 'asset')
    assert hasattr(result, 'side')
    assert hasattr(result, 'qty')
    assert hasattr(result, 'status')