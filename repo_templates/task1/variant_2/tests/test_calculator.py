import pytest
from src.calculator import divide, average, clamp


def test_divide_normal():
    assert divide(10, 2) == 5.0


def test_divide_by_zero():
    # FAILS — ZeroDivisionError because no zero check
    assert divide(10, 0) == 0.0


def test_average_normal():
    assert average([1, 2, 3]) == 2.0


def test_average_empty():
    # FAILS — ZeroDivisionError because empty list not handled
    assert average([]) == 0.0


def test_clamp_within():
    assert clamp(5, 0, 10) == 5


def test_clamp_below():
    assert clamp(-5, 0, 10) == 0


def test_clamp_above():
    assert clamp(15, 0, 10) == 10
