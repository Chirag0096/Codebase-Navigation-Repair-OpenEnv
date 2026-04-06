"""Retry configuration constants."""

MAX_RETRIES = 3              # max number of retry attempts
BASE_DELAY = 0.1             # base delay in seconds
RETRYABLE_EXCEPTIONS = (ConnectionError, TimeoutError, OSError)
