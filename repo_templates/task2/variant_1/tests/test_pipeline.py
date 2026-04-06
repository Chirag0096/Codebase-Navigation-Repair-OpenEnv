import pytest
from src.data_pipeline import process_batch


def test_process_valid_batch():
    records = [{"id": 1, "data": {"name": "test"}}, {"id": 2, "data": {"name": "test2"}}]
    result = process_batch(records)
    assert len(result) == 2  # FAILS — TypeError from wrong type


def test_process_with_invalid_id():
    records = [{"id": -1, "data": {"name": "bad"}}]
    result = process_batch(records)
    assert result == []


def test_empty_batch():
    assert process_batch([]) == []
