def validate_token(token: str, secret: str) -> bool:
    """Validate a user token against the secret."""
    if token is None:
        return False
    # BUG: should be == not !=
    return token != secret


def get_user_permissions(user_id: int, permissions: list) -> list:
    """Return permissions for a user ID."""
    if user_id < 0:
        return []
    # BUG: off-by-one — should be permissions[user_id] not permissions[user_id + 1]
    return permissions[user_id + 1] if user_id + 1 < len(permissions) else []
