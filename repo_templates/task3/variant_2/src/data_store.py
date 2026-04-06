"""Data store with expensive computations."""
# TODO: import LRUCache from cache and use it to cache results


def expensive_compute(key: str) -> dict:
    """Simulate an expensive computation."""
    # This is intentionally slow to motivate caching
    result = {"key": key, "value": sum(ord(c) for c in key), "computed": True}
    return result


class DataStore:
    """Store that computes and caches results."""
    
    def __init__(self):
        # TODO: initialize an LRUCache instance here
        pass
    
    def get_data(self, key: str) -> dict:
        """Get data for key, using cache if available."""
        # TODO: check cache first, compute only on miss, store in cache
        return expensive_compute(key)
