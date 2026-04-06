import pytest
from src.auth import validate_token, get_user_permissions


def test_valid_token():
    assert validate_token("abc123", "abc123") == True  # FAILS because of != bug


def test_invalid_token():
    assert validate_token("wrong", "abc123") == False


def test_none_token():
    assert validate_token(None, "abc123") == False


def test_user_permissions():
    perms = ["read", "write", "admin"]
    assert get_user_permissions(0, perms) == "read"  # FAILS because of off-by-one bug


def test_negative_user_id():
    assert get_user_permissions(-1, ["read"]) == []
