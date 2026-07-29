"""
MySQL access via the mysql_mcp_server MCP server (designcomputer/mysql_mcp_server,
https://github.com/designcomputer/mysql_mcp_server), loaded through
langchain-mcp-adapters and spawned over stdio via `uvx` - no separate
process to manage, uv downloads and runs it on demand.

Version pin note: langchain-mcp-adapters==0.3.1 imports mcp.server.fastmcp,
which was removed in mcp==2.0.0 (a breaking release). pyproject.toml pins
mcp<2.0.0 to keep this working - check that pin before upgrading either
package.

Sync/async note: MultiServerMCPClient.get_tools() returns tools built with
only a `coroutine=` (no sync `func=`), so they raise NotImplementedError if
invoked directly from this project's fully-synchronous agent
(agent.invoke()/agent.stream(), see app/agent/memory.py). _wrap_sync()
bridges each async tool into a plain sync-callable StructuredTool via
asyncio.run() - safe here because these tools are only ever invoked from
a FastAPI sync `def` route running in a worker thread (no event loop
already running on that thread to conflict with), never from inside an
async function.

Read-only note: mysql_mcp_server's `execute_sql` tool allows INSERT/UPDATE/
DELETE too, unlike this project's own sql_tool.py (SELECT-only). Given the
small, tool-call-unreliable local model this project defaults to (see
OLLAMA_MODEL in app/config.py), execute_sql is wrapped with the same
read-only guard as sql_tool.py here - reject anything that isn't
SELECT/SHOW/DESCRIBE/EXPLAIN before it ever reaches the real database.

STUB CONTRACT WARNING: mysql_mcp_server's exact tool names/arg names below
(execute_sql, get_schema_info, get_table_sample; args "query"/"sql") are
taken from its README, not verified against a live server in this
environment (no MySQL/Docker available to test against). Once you point
this at a real database, check the actual tool schemas the agent sees
(e.g. log TOOLS names at startup) and adjust _SQL_ARG_KEYS / tool names
below if they don't match.
"""
import asyncio
import logging

from langchain_core.tools import BaseTool, StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from app.config import MYSQL_DATABASE, MYSQL_HOST, MYSQL_PASSWORD, MYSQL_PORT, MYSQL_USER

logger = logging.getLogger(__name__)

# Tool whose argument is a raw SQL statement - the only one that needs the
# read-only guard. get_schema_info/get_table_sample take table names, not
# SQL, so they're read-only by construction and left unguarded.
_SQL_TOOL_NAME = "execute_sql"
_SQL_ARG_KEYS = ("query", "sql", "statement")
_READ_ONLY_PREFIXES = ("select", "show", "describe", "explain")


def _extract_sql_text(kwargs: dict) -> str | None:
    for key in _SQL_ARG_KEYS:
        value = kwargs.get(key)
        if isinstance(value, str):
            return value
    return None


def _reject_if_not_read_only(kwargs: dict) -> str | None:
    """Returns a rejection message if kwargs holds a non-SELECT/SHOW/
    DESCRIBE/EXPLAIN statement, else None. If the SQL text can't be found
    under a known key, lets the call through unguarded rather than
    silently breaking every query - see STUB CONTRACT WARNING above."""
    sql_text = _extract_sql_text(kwargs)
    if sql_text is None:
        return None
    if not sql_text.strip().lower().startswith(_READ_ONLY_PREFIXES):
        return (
            "Rejected: only read-only queries (SELECT/SHOW/DESCRIBE/EXPLAIN) "
            f"are allowed. Got: {sql_text[:80]!r}"
        )
    return None


def _wrap_sync(async_tool: BaseTool, guard=None) -> BaseTool:
    """Bridges an async-only MCP tool into a plain sync StructuredTool -
    see the sync/async note in this module's docstring for why this is
    needed and why asyncio.run() is safe in this specific call context."""

    def sync_func(**kwargs):
        if guard is not None:
            rejection = guard(kwargs)
            if rejection:
                return rejection
        return asyncio.run(async_tool.ainvoke(kwargs))

    return StructuredTool(
        name=async_tool.name,
        description=async_tool.description,
        args_schema=async_tool.args_schema,
        func=sync_func,
    )


async def _fetch_mysql_tools() -> list[BaseTool]:
    client = MultiServerMCPClient(
        {
            "mysql": {
                "command": "uvx",
                "args": ["mysql_mcp_server"],
                "transport": "stdio",
                "env": {
                    "MYSQL_HOST": MYSQL_HOST,
                    "MYSQL_PORT": MYSQL_PORT,
                    "MYSQL_USER": MYSQL_USER,
                    "MYSQL_PASSWORD": MYSQL_PASSWORD,
                    "MYSQL_DATABASE": MYSQL_DATABASE,
                },
            }
        }
    )
    return await client.get_tools()


def get_mysql_tools() -> list[BaseTool]:
    """
    Loads and sync-wraps the MySQL MCP tools, once, at import time. Fails
    safe: if MYSQL_HOST isn't configured, or the MCP server can't be
    reached (uvx missing, bad credentials, etc.), logs and returns an
    empty list instead of crashing app startup - MySQL support is
    opt-in, same as the inventory API integration.
    """
    if not MYSQL_HOST:
        logger.info("MYSQL_HOST not set - skipping MySQL MCP tool loading.")
        return []

    try:
        async_tools = asyncio.run(_fetch_mysql_tools())
    except Exception:
        logger.exception("Failed to load MySQL MCP tools - continuing without them.")
        return []

    wrapped = []
    for async_tool in async_tools:
        guard = _reject_if_not_read_only if async_tool.name == _SQL_TOOL_NAME else None
        wrapped.append(_wrap_sync(async_tool, guard))

    logger.info("Loaded %d MySQL MCP tool(s): %s", len(wrapped), [t.name for t in wrapped])
    return wrapped
