"""Utility functions for the auth module."""


def sanitize_input(text: str) -> str:
    """Remove leading/trailing whitespace and normalize."""
    if not isinstance(text, str):
        return ""
    return text.strip().lower()


def format_response(status: str, data: dict = None) -> dict:
    """Format a standard API response."""
    return {
        "status": status,
        "data": data or {},
    }
