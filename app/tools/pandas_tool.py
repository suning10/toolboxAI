"""
Lets the agent answer questions over an uploaded spreadsheet/CSV by
writing and executing pandas code, rather than the LLM trying to do
arithmetic itself (which it's unreliable at).

Design choice: one global "active dataframe" set via load_dataset().
Good enough for a single-user/low-volume agent. If you need multiple
concurrent users with different files, key this by session_id instead.
"""
import pandas as pd
from langchain.tools import tool

_active_df: pd.DataFrame | None = None
_active_filename: str | None = None


def load_dataset(filepath: str) -> str:
    """Called by the API layer on file upload, not by the agent itself."""
    global _active_df, _active_filename
    if filepath.endswith(".csv"):
        _active_df = pd.read_csv(filepath)
    elif filepath.endswith((".xlsx", ".xls")):
        _active_df = pd.read_excel(filepath)
    else:
        raise ValueError(f"Unsupported file type: {filepath}")
    _active_filename = filepath
    return f"Loaded {filepath} with shape {_active_df.shape}"


@tool
def describe_dataset() -> str:
    """Get the shape, column names, dtypes, and first few rows of the
    currently loaded dataset. ALWAYS call this first before writing
    pandas queries, so you know the actual column names."""
    if _active_df is None:
        return "No dataset loaded yet. Ask the user to upload a file first."
    info = [
        f"File: {_active_filename}",
        f"Shape: {_active_df.shape}",
        f"Columns and dtypes:\n{_active_df.dtypes.to_string()}",
        f"First 5 rows:\n{_active_df.head().to_string()}",
    ]
    return "\n\n".join(info)


@tool
def run_pandas_query(code: str) -> str:
    """Execute a pandas expression against the loaded dataset and return
    the result. The dataframe is available as variable `df`.

    Example inputs:
    - "df['revenue'].sum()"
    - "df.groupby('region')['sales'].mean().sort_values(ascending=False)"
    - "df[df['status'] == 'overdue'].shape[0]"

    Only use this after calling describe_dataset so column names are
    correct. Do not use import statements or file/network access -
    only operate on the `df` variable.
    """
    if _active_df is None:
        return "No dataset loaded yet. Ask the user to upload a file first."

    # Deliberately restricted execution scope - no builtins, no imports.
    # This is NOT a full sandbox; for a multi-user production deployment
    # run this in a subprocess/container with resource limits instead.
    forbidden = ["import", "__", "open(", "exec(", "eval(", "os.", "sys."]
    if any(f in code for f in forbidden):
        return "Rejected: query contains disallowed operations."

    try:
        result = eval(code, {"__builtins__": {}}, {"df": _active_df, "pd": pd})
        return str(result)
    except Exception as e:
        return f"Error executing query: {e}. Check column names with describe_dataset."
