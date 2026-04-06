"""Logging utilities for inventory operations."""


def log_operation(operation: str, item_id: str, details: str = "") -> str:
    """Create a log entry for an inventory operation."""
    entry = f"[INVENTORY] {operation}: {item_id}"
    if details:
        entry += f" — {details}"
    return entry
