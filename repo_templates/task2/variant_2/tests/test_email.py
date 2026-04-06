import pytest
from src.email_sender import send_welcome_email, send_reset_email


def test_send_welcome_email():
    # FAILS — KeyError because email_sender passes 'name' but template expects 'username'
    result = send_welcome_email("Alice", "alice@example.com")
    assert result["sent"] == True
    assert "Alice" in result["body"]
    assert "alice@example.com" in result["body"]


def test_send_reset_email():
    result = send_reset_email("bob@example.com", "https://reset.link/abc")
    assert result["sent"] == True
    assert "https://reset.link/abc" in result["body"]


def test_welcome_email_structure():
    # FAILS — same KeyError as test_send_welcome_email
    result = send_welcome_email("Charlie", "charlie@test.com")
    assert result["to"] == "charlie@test.com"
    assert result["subject"] == "Welcome!"
