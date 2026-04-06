import pytest
import math
from src.query_builder import Paginator
from src.config import DEFAULT_PAGE_SIZE


def test_paginator_first_page():
    items = list(range(25))
    p = Paginator(items, page=1, page_size=10)
    result = p.get_page()
    assert len(result) == 10
    assert result == list(range(10))


def test_paginator_last_page():
    items = list(range(25))
    p = Paginator(items, page=3, page_size=10)
    result = p.get_page()
    assert len(result) == 5
    assert result == list(range(20, 25))


def test_paginator_total_pages():
    items = list(range(25))
    p = Paginator(items, page=1, page_size=10)
    assert p.total_pages == 3


def test_paginator_has_next():
    items = list(range(25))
    p1 = Paginator(items, page=1, page_size=10)
    assert p1.has_next == True
    p3 = Paginator(items, page=3, page_size=10)
    assert p3.has_next == False


def test_paginator_has_prev():
    items = list(range(25))
    p1 = Paginator(items, page=1, page_size=10)
    assert p1.has_prev == False
    p2 = Paginator(items, page=2, page_size=10)
    assert p2.has_prev == True


def test_paginator_empty():
    p = Paginator([], page=1, page_size=10)
    assert p.get_page() == []
    assert p.total_pages == 0
    assert p.has_next == False


def test_paginator_page_info():
    items = list(range(50))
    p = Paginator(items, page=2, page_size=10)
    info = p.get_page_info()
    assert info["page"] == 2
    assert info["page_size"] == 10
    assert info["total_items"] == 50
    assert info["total_pages"] == 5


def test_paginator_default_page_size():
    items = list(range(50))
    p = Paginator(items, page=1)
    assert len(p.get_page()) == DEFAULT_PAGE_SIZE
