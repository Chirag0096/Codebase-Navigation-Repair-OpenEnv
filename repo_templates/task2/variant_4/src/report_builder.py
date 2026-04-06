"""Report builder that assembles reports with formatted dates."""
from src.date_formatter import format_date_range


def build_monthly_report(title: str, start_date: str, end_date: str, data: list) -> dict:
    """
    Build a monthly report with formatted date header.
    
    Args:
        title: Report title
        start_date: ISO format string 'YYYY-MM-DD'
        end_date: ISO format string 'YYYY-MM-DD'
        data: List of data points
    """
    # BUG: passing ISO string directly, but format_date_range expects datetime objects
    date_header = format_date_range(start_date, end_date)
    
    return {
        "title": title,
        "period": date_header,
        "total_records": len(data),
        "data": data,
    }


def build_summary(title: str, content: str) -> dict:
    """Build a simple summary report."""
    return {"title": title, "content": content, "type": "summary"}
