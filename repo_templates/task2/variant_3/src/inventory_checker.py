"""Inventory checking service. Verifies stock levels for orders."""

# Simulated stock database
STOCK = {
    "WIDGET-A": 100,
    "WIDGET-B": 50,
    "GADGET-X": 0,
    "GADGET-Y": 25,
}


def check_availability(requested_items: dict) -> bool:
    """
    Check if all requested items are available in stock.
    
    Args:
        requested_items: dict mapping SKU to quantity, e.g. {"WIDGET-A": 5, "GADGET-Y": 2}
    
    Returns:
        True if all items are available in sufficient quantity.
    """
    if not isinstance(requested_items, dict):
        raise TypeError(
            f"requested_items must be dict, got {type(requested_items).__name__}. "
            f"Expected format: {{'SKU': quantity}}"
        )
    
    for sku, qty in requested_items.items():
        if sku not in STOCK:
            return False
        if STOCK[sku] < qty:
            return False
    return True
