"""Request handlers for the application."""
# TODO: import validate_input from validators
# TODO: apply @validate_input decorator to each handler


def create_user(data: dict) -> dict:
    """Create a new user from input data.
    Expected fields: name (str), age (int), email (str)
    """
    # TODO: add @validate_input({"name": str, "age": int, "email": str})
    return {
        "id": 1,
        "name": data["name"],
        "age": data["age"],
        "email": data["email"],
        "created": True,
    }


def update_settings(data: dict) -> dict:
    """Update user settings.
    Expected fields: theme (str), notifications (bool)
    """
    # TODO: add @validate_input({"theme": str, "notifications": bool})
    return {
        "theme": data["theme"],
        "notifications": data["notifications"],
        "updated": True,
    }
