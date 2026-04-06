import pytest
from src.cache_manager import CacheManager


def test_cache_set_and_get():
    cache = CacheManager()
    # FAILS — TypeError because bytes passed to serializer instead of str
    cache.set("user:1", "Alice")
    assert cache.get("user:1") == "Alice"


def test_cache_get_missing():
    cache = CacheManager()
    assert cache.get("nonexistent", "default") == "default"


def test_cache_delete():
    cache = CacheManager()
    # FAILS — same TypeError on set
    cache.set("temp", "data")
    assert cache.delete("temp") == True
    assert cache.get("temp") is None


def test_cache_clear():
    cache = CacheManager()
    cache._store["a"] = '{"data": "1"}'
    cache._store["b"] = '{"data": "2"}'
    cache.clear()
    assert cache.keys() == []


def test_cache_keys():
    cache = CacheManager()
    cache._store["x"] = '{"data": "1"}'
    cache._store["y"] = '{"data": "2"}'
    assert sorted(cache.keys()) == ["x", "y"]
