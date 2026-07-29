"""
Test the MySQL MCP tool's read-only guard and sync/async bridging with
a fake async-only tool (mimicking exactly what langchain-mcp-adapters
produces - coroutine= only, no func=) - no real MySQL/MCP server
required. See app/tools/mysql_mcp_tool.py's module docstring for why
the sync bridge is needed at all.

Run: pytest tests/test_mysql_mcp_tool.py -v
"""
import pytest
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.tools import mysql_mcp_tool


class _QueryArgs(BaseModel):
    query: str = Field(...)


def _make_fake_mcp_tool(name="execute_sql"):
    async def call_tool(query: str) -> str:
        return f"[result for: {query}]"

    return StructuredTool(
        name=name,
        description="fake MCP tool",
        args_schema=_QueryArgs,
        coroutine=call_tool,  # async-only, exactly like the real adapter
    )


def test_async_only_tool_rejects_direct_sync_invocation():
    """Confirms the premise the wrapper exists to solve."""
    fake_tool = _make_fake_mcp_tool()
    with pytest.raises(NotImplementedError):
        fake_tool.invoke({"query": "SELECT 1"})


def test_wrap_sync_bridges_async_tool_to_sync_invoke():
    fake_tool = _make_fake_mcp_tool()
    wrapped = mysql_mcp_tool._wrap_sync(fake_tool)
    assert wrapped.invoke({"query": "SELECT 1"}) == "[result for: SELECT 1]"


@pytest.mark.parametrize("query", ["SELECT * FROM orders", "  select 1", "SHOW TABLES", "DESCRIBE orders", "EXPLAIN SELECT 1"])
def test_read_only_guard_allows_read_statements(query):
    fake_tool = _make_fake_mcp_tool()
    wrapped = mysql_mcp_tool._wrap_sync(fake_tool, guard=mysql_mcp_tool._reject_if_not_read_only)
    result = wrapped.invoke({"query": query})
    assert result.startswith("[result for:")


@pytest.mark.parametrize("query", ["DELETE FROM orders", "UPDATE orders SET x=1", "INSERT INTO orders VALUES (1)", "DROP TABLE orders"])
def test_read_only_guard_rejects_write_statements(query):
    fake_tool = _make_fake_mcp_tool()
    wrapped = mysql_mcp_tool._wrap_sync(fake_tool, guard=mysql_mcp_tool._reject_if_not_read_only)
    result = wrapped.invoke({"query": query})
    assert result.startswith("Rejected:")
    assert "[result for:" not in result


def test_get_mysql_tools_skips_when_host_not_set(monkeypatch):
    monkeypatch.setattr(mysql_mcp_tool, "MYSQL_HOST", "")
    assert mysql_mcp_tool.get_mysql_tools() == []


def test_get_mysql_tools_fails_safe_on_connection_error(monkeypatch):
    monkeypatch.setattr(mysql_mcp_tool, "MYSQL_HOST", "some-host")

    async def boom():
        raise RuntimeError("uvx not found / connection refused")

    monkeypatch.setattr(mysql_mcp_tool, "_fetch_mysql_tools", boom)
    assert mysql_mcp_tool.get_mysql_tools() == []
