"""Helper utilities for the calculator module."""


def parse_number(value: str) -> float:
    """Parse a string to a float, returning 0.0 on failure."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0


def format_result(value: float, decimals: int = 2) -> str:
    """Format a numeric result to a string with given decimal places."""
    return f"{value:.{decimals}f}"
