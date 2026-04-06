"""API endpoints that return paginated results."""
# TODO: import Paginator from query_builder
from src.query_builder import build_query


def list_items(table: str, page: int = 1, page_size: int = None) -> dict:
    """List items from a table with pagination."""
    # TODO: use Paginator to paginate results
    items = build_query(table)
    return {
        "items": items,  # Should return only current page
        "total": len(items),
    }
