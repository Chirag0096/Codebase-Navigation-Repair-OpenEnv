"""HTTP client with retry capabilities."""
# TODO: implement RetryHandler class
# Must have: execute(func, *args, **kwargs) method
# Uses retry_config.MAX_RETRIES, BASE_DELAY, RETRYABLE_EXCEPTIONS
# Should use exponential backoff: wait = base_delay * (2 ** attempt)
# Should store attempt logs in self.attempts list
