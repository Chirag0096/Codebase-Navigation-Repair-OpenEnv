import pytest
from datetime import datetime, timedelta
from src.scheduler import is_available, get_next_available, days_until


def test_slot_available():
    booked = [
        {"start": datetime(2024, 1, 1, 10, 0), "end": datetime(2024, 1, 1, 11, 0)}
    ]
    assert is_available(
        datetime(2024, 1, 1, 12, 0),
        datetime(2024, 1, 1, 13, 0),
        booked
    ) == True


def test_slot_overlap():
    booked = [
        {"start": datetime(2024, 1, 1, 10, 0), "end": datetime(2024, 1, 1, 11, 0)}
    ]
    assert is_available(
        datetime(2024, 1, 1, 10, 30),
        datetime(2024, 1, 1, 11, 30),
        booked
    ) == False


def test_adjacent_slots_allowed():
    """Meeting starting exactly when another ends should be allowed."""
    booked = [
        {"start": datetime(2024, 1, 1, 10, 0), "end": datetime(2024, 1, 1, 11, 0)}
    ]
    # FAILS — returns False because <= is used instead of <
    assert is_available(
        datetime(2024, 1, 1, 11, 0),
        datetime(2024, 1, 1, 12, 0),
        booked
    ) == True


def test_days_until():
    now = datetime(2024, 1, 1, 0, 0)
    target = datetime(2024, 1, 11, 0, 0)
    # FAILS — returns 11 instead of 10 because of +1 bug
    assert days_until(target, now) == 10


def test_days_until_same_day():
    now = datetime(2024, 6, 15, 8, 0)
    target = datetime(2024, 6, 15, 20, 0)
    # FAILS — returns 1 instead of 0
    assert days_until(target, now) == 0
