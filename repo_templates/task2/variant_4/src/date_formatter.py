"""Date formatting utilities for reports."""
from datetime import datetime


def format_date_range(start: datetime, end: datetime) -> str:
    """
    Format a date range for display in reports.
    
    Args:
        start: datetime object for range start
        end: datetime object for range end
    
    Returns:
        Formatted string like "Jan 01, 2024 — Jan 31, 2024"
    """
    if not isinstance(start, datetime):
        raise TypeError(f"start must be datetime, got {type(start).__name__}")
    if not isinstance(end, datetime):
        raise TypeError(f"end must be datetime, got {type(end).__name__}")
    
    return f"{start.strftime('%b %d, %Y')} — {end.strftime('%b %d, %Y')}"


def format_single_date(dt: datetime) -> str:
    """Format a single date."""
    if not isinstance(dt, datetime):
        raise TypeError(f"Expected datetime, got {type(dt).__name__}")
    return dt.strftime("%B %d, %Y")
