# Feature: Add Pagination Support to Query Builder

## Background
The query builder in src/query_builder.py constructs database queries but currently returns all results. We need pagination.

## What to implement
Add a `Paginator` class in src/query_builder.py that:
1. Takes `items` (list), `page` (int, 1-indexed), and `page_size` (int) from config
2. Implements `get_page()` -> returns the items for the current page
3. Implements `total_pages` property -> returns total number of pages
4. Implements `has_next` property -> True if there are more pages
5. Implements `has_prev` property -> True if current page > 1
6. Implements `get_page_info()` -> returns dict with page metadata

Then use Paginator in src/api_endpoints.py to paginate query results.

## Files to modify
- src/query_builder.py — implement Paginator class
- src/api_endpoints.py — use Paginator in list_items endpoint

## Do not modify
- tests/test_pagination.py — tests are already written, make them pass
- src/config.py — contains DEFAULT_PAGE_SIZE
