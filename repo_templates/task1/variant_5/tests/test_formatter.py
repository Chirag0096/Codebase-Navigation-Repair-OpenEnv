import pytest
from src.formatter import truncate, extract_between, capitalize_words


def test_truncate_short():
    assert truncate("hello", 10) == "hello"


def test_truncate_long():
    # FAILS — returns "hello worl..." (13 chars) instead of "hello w..." (10 chars)
    result = truncate("hello world", 10)
    assert len(result) <= 10
    assert result == "hello w..."


def test_truncate_empty():
    assert truncate("", 5) == ""


def test_extract_between():
    text = "start[CONTENT]end"
    # FAILS — returns "[CONTENT]" instead of "CONTENT" because start_idx not offset
    assert extract_between(text, "[", "]") == "CONTENT"


def test_extract_missing_marker():
    assert extract_between("no markers here", "[", "]") == ""


def test_capitalize_words():
    assert capitalize_words("hello world foo") == "Hello World Foo"


def test_capitalize_single():
    assert capitalize_words("test") == "Test"
