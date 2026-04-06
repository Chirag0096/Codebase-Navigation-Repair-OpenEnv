import pytest
from src.handlers import create_user, update_settings


def test_create_user_valid():
    result = create_user({"name": "Alice", "age": 30, "email": "alice@test.com"})
    assert result["created"] == True
    assert result["name"] == "Alice"


def test_create_user_missing_field():
    with pytest.raises(ValueError, match="missing"):
        create_user({"name": "Bob"})


def test_create_user_wrong_type():
    with pytest.raises(ValueError, match="type"):
        create_user({"name": "Charlie", "age": "thirty", "email": "c@test.com"})


def test_update_settings_valid():
    result = update_settings({"theme": "dark", "notifications": True})
    assert result["updated"] == True


def test_update_settings_missing():
    with pytest.raises(ValueError, match="missing"):
        update_settings({"theme": "light"})


def test_update_settings_wrong_type():
    with pytest.raises(ValueError, match="type"):
        update_settings({"theme": 123, "notifications": True})
