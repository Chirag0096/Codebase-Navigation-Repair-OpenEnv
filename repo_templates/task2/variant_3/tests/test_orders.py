import pytest
from src.order_processor import process_order


def test_process_valid_order():
    order = {
        "items": [{"sku": "WIDGET-A", "qty": 2}, {"sku": "GADGET-Y", "qty": 1}],
        "customer": "alice@example.com",
    }
    # FAILS — TypeError because list is passed instead of dict
    result = process_order(order)
    assert result["status"] == "confirmed"


def test_empty_order():
    result = process_order({"items": [], "customer": "bob@example.com"})
    assert result["status"] == "error"


def test_order_structure():
    order = {
        "items": [{"sku": "WIDGET-B", "qty": 5}],
        "customer": "charlie@example.com",
    }
    # FAILS — same TypeError
    result = process_order(order)
    assert "items" in result
