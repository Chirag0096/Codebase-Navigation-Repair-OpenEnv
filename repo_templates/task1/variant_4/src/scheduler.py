"""Meeting and event scheduler module."""
from datetime import datetime, timedelta


def is_available(start: datetime, end: datetime, booked_slots: list) -> bool:
    """Check if a time slot is available (no overlap with booked slots)."""
    for slot in booked_slots:
        slot_start = slot["start"]
        slot_end = slot["end"]
        # BUG: off-by-one — should be < not <= for end comparison
        # Adjacent meetings (one ends exactly when another starts) should be allowed
        if start <= slot_end and end >= slot_start:
            return False
    return True


def get_next_available(after: datetime, duration_minutes: int, booked_slots: list) -> datetime:
    """Find the next available slot after the given time."""
    candidate = after
    for _ in range(100):  # safety limit
        candidate_end = candidate + timedelta(minutes=duration_minutes)
        if is_available(candidate, candidate_end, booked_slots):
            return candidate
        candidate += timedelta(minutes=15)  # check in 15-minute increments
    return None


def days_until(target: datetime, now: datetime = None) -> int:
    """Calculate whole days until target date."""
    if now is None:
        now = datetime.now()
    delta = target - now
    # BUG: should return delta.days, not delta.days + 1
    return delta.days + 1
