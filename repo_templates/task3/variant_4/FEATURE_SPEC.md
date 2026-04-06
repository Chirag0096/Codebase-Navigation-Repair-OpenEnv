# Feature: Add HTTP Retry Mechanism

## Background
The HTTP client in src/http_client.py makes external API calls that sometimes fail with transient errors (5xx, timeouts). We need automatic retry logic.

## What to implement
Create a retry decorator/function in src/http_client.py that:
1. Retries on specified exception types (from retry_config.py)
2. Uses exponential backoff: wait = base_delay * (2 ** attempt)
3. Respects MAX_RETRIES from config
4. Raises the last exception if all retries are exhausted
5. Keeps a log of attempts (list of attempt dicts)

## Files to modify
- src/http_client.py — implement `RetryHandler` class with `execute(func, *args, **kwargs)` method

## Do not modify
- tests/test_retry.py — tests are already written, make them pass
- src/retry_config.py — contains retry configuration
