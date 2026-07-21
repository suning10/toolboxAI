"""
Test the inventory tool in isolation with the external API call mocked
out - same rationale as test_tools.py: catch bugs before debugging
through the agent's tool-calling loop.

Run: pytest tests/test_inventory_tool.py -v
"""
import httpx
import pytest
from app.tools import inventory_tool


class _FakeResponse:
    def __init__(self, json_data, status_code=200):
        self._json = json_data
        self.status_code = status_code
        self.text = str(json_data)

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("GET", "http://testserver")
            raise httpx.HTTPStatusError("error", request=request, response=self)

    def json(self):
        return self._json


def test_check_inventory_gap_success(monkeypatch):
    def fake_get(url, params=None, headers=None, timeout=None):
        assert params == {"date": "2026-07-16", "sloc": "1001"}
        return _FakeResponse({"sloc": "1001", "date": "2026-07-16", "gap_qty": -12, "unit": "EA"})

    monkeypatch.setattr(inventory_tool.httpx, "get", fake_get)
    result = inventory_tool.check_inventory_gap.invoke({"date": "2026-07-16", "sloc": "1001"})
    assert "SLOC 1001" in result
    assert "-12" in result
    assert "EA" in result


def test_check_inventory_gap_invalid_date():
    result = inventory_tool.check_inventory_gap.invoke({"date": "not-a-date", "sloc": "1001"})
    assert "Invalid date" in result


def test_check_inventory_gap_missing_sloc():
    result = inventory_tool.check_inventory_gap.invoke({"date": "2026-07-16", "sloc": ""})
    assert "Missing sloc" in result


def test_check_inventory_gap_api_error(monkeypatch):
    def fake_get(url, params=None, headers=None, timeout=None):
        return _FakeResponse({"detail": "not found"}, status_code=404)

    monkeypatch.setattr(inventory_tool.httpx, "get", fake_get)
    result = inventory_tool.check_inventory_gap.invoke({"date": "2026-07-16", "sloc": "9999"})
    assert "Inventory API error" in result

def test_check_inventory_gap_missing_sloc():
    result = inventory_tool.check_inventory_gap.invoke({"date": "2026-07-16", "sloc": "WC1E"})
    assert "Missing sloc" in result
