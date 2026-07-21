"""
Checks the inventory gap (expected vs. actual stock variance) for a
storage location (SLOC) on a given date by calling an external
inventory system's REST API - the API computes the gap itself, this
tool just calls it and formats the result for the agent.

STUB CONTRACT: the endpoint path, auth header, and response field
names below are placeholders (see INVENTORY_API_BASE_URL in
app/config.py). Update them to match the real inventory API once you
have its docs:
  GET {INVENTORY_API_BASE_URL}/inventory/gap?date=YYYY-MM-DD&sloc=<code>
  Header: Authorization: Bearer <INVENTORY_API_KEY>
  Response: {"sloc": "...", "date": "...", "gap_qty": <number>, "unit": "..."}
"""
from datetime import date as _date, datetime

import httpx
from langchain.tools import tool

from app.config import INVENTORY_API_BASE_URL, INVENTORY_API_KEY


@tool
def check_inventory_gap(date: str, sloc: str) -> str:
    """Check the inventory gap (expected vs. actual stock variance) for
    a specific storage location on a specific date. User May also called it SCR(stock comparison report)

    Args:
        date: ISO date string, e.g. "2026-07-16".
        sloc: Storage location code, e.g. "1001".
    """
    try:
        parsed_date = _date.fromisoformat(date)
        daydiff = parsed_date - _date.fromisoformat(datetime.now().strftime('%Y-%m-%d'))
    except ValueError:
        return f"Invalid date '{date}'. Use ISO format, e.g. 2026-07-16."

    if not sloc:
        return "Missing sloc: provide a storage location code."

    try:
        response = httpx.get(
            f"{INVENTORY_API_BASE_URL}?date={date}",
            # params={"date": parsed_date.isoformat(), "sloc": sloc},
            headers={"Token": f"{INVENTORY_API_KEY}"},
            timeout=15.0,
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        return f"Inventory API error ({e.response.status_code}): {e.response.text}"
    except httpx.HTTPError as e:
        return f"Could not reach inventory API: {e}"

    data = response.json()
    gap = data.get("data")[-1].get("absoluteGap")
    if gap is None:
        return f"Inventory API returned no gap value for sloc={sloc} on {date}: {data}"

    # unit = data.get("unit", "")
    return f"Inventory gap for SLOC {sloc} on {date}: {gap}".strip()
    # return f"Inventory gap for SLOC {sloc} on {date}: 100".strip()
