# Feature: Add request rate limiting to the API

## Background
The current API in src/api.py has no rate limiting. Any client can make unlimited requests.

## What to implement
Add a rate limiter that:
1. Tracks requests per client IP in a dict stored in src/middleware.py
2. Allows maximum 5 requests per minute per IP
3. Returns HTTP 429 status with message "Rate limit exceeded" when limit is hit
4. Resets the count for an IP after 60 seconds of no requests

## Files to modify
- src/middleware.py — add RateLimiter class with is_allowed(ip: str) -> bool method
- src/api.py — import and use RateLimiter in the request handler

## Do not modify
- tests/test_api.py — tests are already written, make them pass
- src/config.py — contains RATE_LIMIT and RATE_WINDOW constants you should use
