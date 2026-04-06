"""Text formatter module for processing and formatting strings."""


def truncate(text: str, max_length: int) -> str:
    """Truncate text to max_length, adding '...' if truncated."""
    if not text:
        return ""
    if len(text) <= max_length:
        return text
    # BUG: should be text[:max_length - 3] + "..." to account for ellipsis length
    return text[:max_length] + "..."


def extract_between(text: str, start_marker: str, end_marker: str) -> str:
    """Extract text between two markers."""
    start_idx = text.find(start_marker)
    if start_idx == -1:
        return ""
    # BUG: should start after the marker, i.e. start_idx + len(start_marker)
    content_start = start_idx  # wrong — includes the start_marker itself
    end_idx = text.find(end_marker, content_start)
    if end_idx == -1:
        return ""
    return text[content_start:end_idx]


def capitalize_words(text: str) -> str:
    """Capitalize the first letter of every word."""
    return " ".join(w.capitalize() for w in text.split())
