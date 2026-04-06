"""Cache management service that stores serialized data."""
from src.serializer import serialize_value, deserialize_value


class CacheManager:
    """Simple in-memory cache with serialization."""
    
    def __init__(self):
        self._store = {}
    
    def set(self, key: str, value) -> None:
        """Store a value in the cache after serializing it."""
        # BUG: passing bytes (encoded) instead of str to serialize_value
        serialized = serialize_value(str(value).encode('utf-8'))
        self._store[key] = serialized
    
    def get(self, key: str, default=None):
        """Retrieve and deserialize a value from cache."""
        if key not in self._store:
            return default
        return deserialize_value(self._store[key])
    
    def delete(self, key: str) -> bool:
        """Remove a key from cache."""
        if key in self._store:
            del self._store[key]
            return True
        return False
    
    def clear(self):
        """Clear all cached values."""
        self._store.clear()
    
    def keys(self) -> list:
        """Return all cache keys."""
        return list(self._store.keys())
