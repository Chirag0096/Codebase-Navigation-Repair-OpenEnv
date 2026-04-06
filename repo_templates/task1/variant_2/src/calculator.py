"""Calculator module with basic math operations."""


def divide(numerator: float, denominator: float) -> float:
    """Divide numerator by denominator safely."""
    # BUG: missing zero-division check — should check denominator == 0
    return numerator / denominator


def average(numbers: list) -> float:
    """Calculate the average of a list of numbers."""
    # BUG: doesn't handle empty list — should return 0.0 for empty
    total = sum(numbers)
    return total / len(numbers)


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value between min and max."""
    if value < min_val:
        return min_val
    if value > max_val:
        return max_val
    return value
