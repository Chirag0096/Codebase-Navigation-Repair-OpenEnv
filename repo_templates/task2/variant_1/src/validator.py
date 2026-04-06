def validate_record(record_id: int, data: dict) -> dict:
    """Validate a record. record_id must be a positive integer."""
    if not isinstance(record_id, int):
        raise TypeError(f"record_id must be int, got {type(record_id)}")
    if record_id <= 0:
        return None
    return {"id": record_id, "data": data, "valid": True}
