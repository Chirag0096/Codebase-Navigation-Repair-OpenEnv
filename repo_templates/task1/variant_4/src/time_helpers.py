"""Time helper functions."""
from datetime import datetime


def format_time(dt: datetime) -> str:
    """Format datetime to string."""
    return dt.strftime("%Y-%m-%d %H:%M")


def parse_time(s: str) -> datetime:
    """Parse string to datetime."""
    return datetime.strptime(s, "%Y-%m-%d %H:%M")
