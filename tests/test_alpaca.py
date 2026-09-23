"""Tests for Alpaca broker integration (T7)."""
from __future__ import annotations

from hermes_bot.execution import Order
from hermes_bot.execution.brokers.alpaca import AlpacaBroker


def test_alpaca_broker_creation() -> None:
    """Test that AlpacaBroker is created correctly."""
    broker = AlpacaBroker()
    
    # Check that it is a BaseBroker
    assert hasattr(broker, 'submit')
    assert hasattr(broker, 'get_positions')
    assert hasattr(broker, 'health')


def test_alpaca_broker_health() -> None:
    """Test dat health() werkt."""
    broker = AlpacaBroker()
    
    # Test that the health method works
    health_status = broker.health()
    assert isinstance(health_status, bool)


def test_alpaca_broker_submit() -> None:
    """Test dat submit() werkt."""
    broker = AlpacaBroker()
    
    # Create a dummy order
    order = Order(asset="AAPL", side="buy", qty=10.0)
    
    # Test submit
    result = broker.submit(order)
    
    # Check that the result is correct
    assert hasattr(result, 'asset')
    assert hasattr(result, 'side')
    assert hasattr(result, 'qty')
    assert hasattr(result, 'status')