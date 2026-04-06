import pytest
from src.cache import LRUCache
from src.config import CACHE_MAX_SIZE


def test_cache_put_and_get():
    cache = LRUCache(max_size=CACHE_MAX_SIZE)
    cache.put("a", 1)
    assert cache.get("a") == 1


def test_cache_miss():
    cache = LRUCache(max_size=CACHE_MAX_SIZE)
    assert cache.get("nonexistent") is None


def test_cache_eviction():
    cache = LRUCache(max_size=3)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)
    cache.put("d", 4)  # should evict "a"
    assert cache.get("a") is None
    assert cache.get("d") == 4


def test_cache_lru_order():
    cache = LRUCache(max_size=3)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)
    cache.get("a")  # access "a" makes it most recent
    cache.put("d", 4)  # should evict "b" (least recently used)
    assert cache.get("a") == 1
    assert cache.get("b") is None


def test_cache_size():
    cache = LRUCache(max_size=5)
    assert cache.size == 0
    cache.put("x", 10)
    cache.put("y", 20)
    assert cache.size == 2


def test_cache_clear():
    cache = LRUCache(max_size=5)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.clear()
    assert cache.size == 0
    assert cache.get("a") is None


def test_cache_update_existing():
    cache = LRUCache(max_size=3)
    cache.put("a", 1)
    cache.put("a", 100)
    assert cache.get("a") == 100
    assert cache.size == 1
