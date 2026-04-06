from src.validator import validate_record


def process_batch(records: list) -> list:
    """Process a batch of records through the validation pipeline."""
    results = []
    for record in records:
        # BUG: passing record["id"] as string, but validate_record expects int
        validated = validate_record(str(record["id"]), record["data"])
        if validated:
            results.append(validated)
    return results
