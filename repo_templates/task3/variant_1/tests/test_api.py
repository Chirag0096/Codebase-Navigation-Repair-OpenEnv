import pytest
import time
from unittest.mock import patch
from src.middleware import RateLimiter
from src.config import RATE_LIMIT


def test_rate_limiter_allows_within_limit():
    rl = RateLimiter()
    for _ in range(RATE_LIMIT):
        assert rl.is_allowed("192.168.1.1") == True


def test_rate_limiter_blocks_over_limit():
    rl = RateLimiter()
    for _ in range(RATE_LIMIT):
        rl.is_allowed("192.168.1.1")
    assert rl.is_allowed("192.168.1.1") == False


def test_rate_limiter_different_ips():
    rl = RateLimiter()
    for _ in range(RATE_LIMIT):
        rl.is_allowed("192.168.1.1")
    # Different IP should still be allowed
    assert rl.is_allowed("10.0.0.1") == True


def test_rate_limiter_resets_after_window():
    rl = RateLimiter()
    with patch('time.time') as mock_time:
        mock_time.return_value = 1000.0
        for _ in range(RATE_LIMIT):
            rl.is_allowed("192.168.1.1")
        assert rl.is_allowed("192.168.1.1") == False
        # Advance time past window
        mock_time.return_value = 1065.0
        assert rl.is_allowed("192.168.1.1") == True
