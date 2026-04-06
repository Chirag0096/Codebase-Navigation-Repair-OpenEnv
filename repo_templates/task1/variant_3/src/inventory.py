"""Inventory management module."""


def check_stock(item_id: str, inventory: dict) -> bool:
    """Check if an item is in stock (quantity > 0)."""
    if item_id not in inventory:
        return False
    # BUG: should be > 0, not >= 0 (zero stock means out of stock)
    return inventory[item_id] >= 0


def restock(item_id: str, quantity: int, inventory: dict) -> dict:
    """Add stock for an item."""
    if quantity < 0:
        raise ValueError("Cannot restock negative quantity")
    if item_id in inventory:
        inventory[item_id] += quantity
    else:
        inventory[item_id] = quantity
    return inventory


def get_low_stock_items(inventory: dict, threshold: int = 5) -> list:
    """Return items with stock below threshold."""
    # BUG: should be < threshold, not <= threshold
    return [item for item, qty in inventory.items() if qty <= threshold]
