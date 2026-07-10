"""
Test tools in isolation BEFORE wiring them into the agent - this is
the fastest way to catch bugs, since debugging a tool through the LLM's
tool-calling loop is much slower than calling it directly.

Run: pytest tests/test_tools.py -v
"""
import pandas as pd
import pytest
from app.tools import pandas_tool


@pytest.fixture
def sample_csv(tmp_path):
    df = pd.DataFrame({
        "region": ["East", "West", "East", "West"],
        "sales": [100, 200, 150, 250],
    })
    filepath = tmp_path / "sample.csv"
    df.to_csv(filepath, index=False)
    return str(filepath)


def test_load_and_describe(sample_csv):
    pandas_tool.load_dataset(sample_csv)
    result = pandas_tool.describe_dataset.invoke({})
    assert "region" in result
    assert "sales" in result


def test_run_pandas_query(sample_csv):
    pandas_tool.load_dataset(sample_csv)
    result = pandas_tool.run_pandas_query.invoke({"code": "df['sales'].sum()"})
    assert result == "700"


def test_rejects_disallowed_query(sample_csv):
    pandas_tool.load_dataset(sample_csv)
    result = pandas_tool.run_pandas_query.invoke({"code": "import os"})
    assert "Rejected" in result
