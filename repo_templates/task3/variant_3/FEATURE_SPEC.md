# Feature: Add Input Validation Decorator

## Background
The handlers in src/handlers.py accept user input but have no validation. We need a reusable decorator.

## What to implement
Create a `validate_input` decorator in src/validators.py that:
1. Takes a schema dict mapping field names to types (e.g., `{"name": str, "age": int}`)
2. Validates the first argument (a dict) passed to the decorated function
3. Raises `ValueError` with descriptive message if:
   - A required field is missing from the input
   - A field has the wrong type
4. Passes through to the original function if validation succeeds

Then apply the decorator to handlers in src/handlers.py.

## Files to modify
- src/validators.py — implement the `validate_input` decorator
- src/handlers.py — apply `@validate_input(schema)` to each handler

## Do not modify
- tests/test_handlers.py — tests are already written, make them pass
- src/config.py — contains handler configuration
