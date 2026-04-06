import pytest
from src.inventory import check_stock, restock, get_low_stock_items


def test_in_stock():
    inv = {"apple": 10, "banana": 5}
    assert check_stock("apple", inv) == True


def test_out_of_stock():
    inv = {"apple": 0}
    # FAILS — returns True because >= 0 is wrong, should be > 0
    assert check_stock("apple", inv) == False


def test_item_not_found():
    assert check_stock("ghost", {}) == False


def test_restock_existing():
    inv = {"apple": 5}
    result = restock("apple", 3, inv)
    assert result["apple"] == 8


def test_restock_new():
    inv = {}
    result = restock("orange", 10, inv)
    assert result["orange"] == 10


def test_restock_negative():
    with pytest.raises(ValueError):
        restock("apple", -1, {})


def test_low_stock_items():
    inv = {"apple": 3, "banana": 5, "cherry": 10}
    # FAILS — banana (qty=5) should NOT be in low stock when threshold=5
    # but <= threshold incorrectly includes items AT the threshold
    result = get_low_stock_items(inv, threshold=5)
    assert "apple" in result
    assert "banana" not in result
    assert "cherry" not in result
