"""Query builder with pagination support."""
# TODO: implement Paginator class
# Must have: get_page(), total_pages, has_next, has_prev, get_page_info()
# Uses config.DEFAULT_PAGE_SIZE


def build_query(table: str, filters: dict = None) -> list:
    """Build and execute a mock query, returning all matching items."""
    # Simulated database results
    all_items = [{"id": i, "table": table, "data": f"item_{i}"} for i in range(1, 51)]
    
    if filters:
        # Simple filter simulation
        for key, value in filters.items():
            all_items = [item for item in all_items if item.get(key) == value]
    
    return all_items
