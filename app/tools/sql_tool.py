"""
SQL tool - useful when the agent (or the user's phrasing) reasons more
naturally in SQL than pandas, e.g. multi-table joins later on. For now
it mirrors the same uploaded dataset into an in-memory SQLite table.
"""
import sqlite3
import pandas as pd
from langchain.tools import tool

_conn: sqlite3.Connection | None = None
_table_name = "data"


def load_dataset_sql(filepath: str) -> str:
    """Mirrors the uploaded file into an in-memory SQLite table."""
    global _conn
    if filepath.endswith(".csv"):
        df = pd.read_csv(filepath)
    elif filepath.endswith((".xlsx", ".xls")):
        df = pd.read_excel(filepath)
    else:
        raise ValueError(f"Unsupported file type: {filepath}")

    _conn = sqlite3.connect(":memory:", check_same_thread=False)
    df.to_sql(_table_name, _conn, index=False, if_exists="replace")
    return f"Loaded into SQL table '{_table_name}' with {len(df)} rows"


@tool
def run_sql_query(query: str) -> str:
    """Run a read-only SQL SELECT query against the table named 'data'
    (the currently uploaded dataset). Use describe_dataset first to see
    column names before writing SQL.

    Example: "SELECT region, AVG(sales) FROM data GROUP BY region"
    """
    if _conn is None:
        return "No dataset loaded yet. Ask the user to upload a file first."

    stripped = query.strip().lower()
    if not stripped.startswith("select"):
        return "Rejected: only SELECT queries are allowed."

    try:
        result = pd.read_sql_query(query, _conn)
        return result.to_string(index=False)
    except Exception as e:
        return f"SQL error: {e}. Check column names with describe_dataset."
