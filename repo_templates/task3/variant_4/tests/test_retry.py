import pytest
from unittest.mock import patch
from src.http_client import RetryHandler
from src.retry_config import MAX_RETRIES


def test_execute_success_first_try():
    handler = RetryHandler()
    result = handler.execute(lambda: "ok")
    assert result == "ok"
    assert len(handler.attempts) == 1


def test_execute_retries_on_failure():
    call_count = 0
    def flaky():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ConnectionError("connection failed")
        return "success"
    
    handler = RetryHandler()
    with patch('time.sleep'):  # skip actual delays
        result = handler.execute(flaky)
    assert result == "success"
    assert len(handler.attempts) == 3


def test_execute_exhausts_retries():
    def always_fail():
        raise ConnectionError("permanent failure")
    
    handler = RetryHandler()
    with patch('time.sleep'):
        with pytest.raises(ConnectionError):
            handler.execute(always_fail)
    assert len(handler.attempts) == MAX_RETRIES + 1


def test_non_retryable_exception():
    def bad_input():
        raise ValueError("bad input — not retryable")
    
    handler = RetryHandler()
    with pytest.raises(ValueError):
        handler.execute(bad_input)
    assert len(handler.attempts) == 1


def test_exponential_backoff_delays():
    call_count = 0
    def fail_twice():
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            raise TimeoutError("timeout")
        return "done"
    
    handler = RetryHandler()
    delays = []
    with patch('time.sleep', side_effect=lambda d: delays.append(d)):
        handler.execute(fail_twice)
    
    # Should have exponential delays
    assert len(delays) == 2
    assert delays[1] > delays[0]
