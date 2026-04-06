"""Serialization utilities for the cache system."""
import json


def serialize_value(value: str) -> str:
    """
    Serialize a value to a JSON string for storage.
    
    Args:
        value: must be a string (str type)
    
    Returns:
        JSON-encoded string
    """
    if not isinstance(value, str):
        raise TypeError(f"value must be str, got {type(value).__name__}")
    return json.dumps({"data": value})


def deserialize_value(serialized: str):
    """Deserialize a JSON string back to the original value."""
    if not isinstance(serialized, str):
        raise TypeError(f"serialized must be str, got {type(serialized).__name__}")
    result = json.loads(serialized)
    return result.get("data")
