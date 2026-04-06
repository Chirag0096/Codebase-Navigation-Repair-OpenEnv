import pytest
from src.report_builder import build_monthly_report, build_summary


def test_build_monthly_report():
    # FAILS — TypeError because ISO string passed instead of datetime
    result = build_monthly_report(
        "Sales Report",
        "2024-01-01",
        "2024-01-31",
        [{"amount": 100}, {"amount": 200}],
    )
    assert result["title"] == "Sales Report"
    assert result["total_records"] == 2
    assert "Jan" in result["period"]


def test_build_summary():
    result = build_summary("Q1 Summary", "Revenue increased 15%")
    assert result["title"] == "Q1 Summary"
    assert result["type"] == "summary"


def test_report_structure():
    # FAILS — same TypeError
    result = build_monthly_report("Inventory", "2024-03-01", "2024-03-31", [])
    assert "period" in result
    assert result["total_records"] == 0
