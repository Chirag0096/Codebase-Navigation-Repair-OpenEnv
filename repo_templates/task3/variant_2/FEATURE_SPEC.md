# Feature: Implement an LRU Cache

## Background
The data store in src/data_store.py performs expensive computations. We need a Least Recently Used (LRU) cache to avoid redundant calculations.

## What to implement
Add an LRU cache class in src/cache.py that:
1. Has a configurable `max_size` parameter (use CACHE_MAX_SIZE from config)
2. Implements `get(key)` -> returns value or None if not cached
3. Implements `put(key, value)` -> stores value, evicts LRU entry if at capacity
4. Implements `size` property -> returns current number of cached items
5. Implements `clear()` -> removes all entries

Then integrate it in src/data_store.py by caching compute results.

## Files to modify
- src/cache.py — implement the LRUCache class
- src/data_store.py — use LRUCache to cache expensive_compute results

## Do not modify
- tests/test_cache.py — tests are already written, make them pass
- src/config.py — contains CACHE_MAX_SIZE constant
