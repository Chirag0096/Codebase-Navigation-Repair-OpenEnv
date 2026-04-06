"""Order processing module that checks inventory before fulfillment."""
from src.inventory_checker import check_availability


def process_order(order: dict) -> dict:
    """
    Process an order by checking inventory availability.
    order format: {"items": [{"sku": "ABC", "qty": 2}, ...], "customer": "..."}
    """
    items = order.get("items", [])
    if not items:
        return {"status": "error", "message": "No items in order"}

    # BUG: passing items as list, but check_availability expects a dict {sku: qty}
    available = check_availability(items)

    if available:
        return {"status": "confirmed", "items": items}
    else:
        return {"status": "out_of_stock", "items": items}
