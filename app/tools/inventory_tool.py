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
import logging
from datetime import date as _date, datetime, date, timedelta

import httpx
from langchain.tools import tool

from app.config import INVENTORY_API_BASE_URL, INVENTORY_API_KEY

logger = logging.getLogger(__name__)


@tool
def check_inventory_gap(day: str, sloc= "") -> str:
    """Check the inventory gap (expected vs. actual stock variance) for
    a specific storage location on a specific date(optional). User May also called it SCR(stock comparison report)
    the data returned from API call may have all sloc and total, pick the one user asked.
    if sloc is empty, return total
    do not use this tool if user provide a sku, a sku is usually 9 - 14 chars, eg what is the scr gap for sku SM-T220NZSAXAR in wc1e
    do not use this tool if user ask for detail or what cause the gap

    Args:
        day: ISO date string, e.g. "2026-07-18".
        sloc: Storage location code, sloc can only be WC1E, WR2E, WR3E, WC8E

    """

    try:
        parsed_date = _date.fromisoformat(day)
        daydiff:timedelta = date.today() - parsed_date
    except ValueError:
        return f"Invalid date '{day}'. Use ISO format, e.g. 2026-07-16."

    # if not sloc:
    #     return "Missing sloc: provide a storage location code."
    # logger.info("Inventory API request to %s succeeded", INVENTORY_API_BASE_URL)
    try:
        response = httpx.get(
            f"{INVENTORY_API_BASE_URL}/scrReportSummary?date={daydiff.days}",
            # params={"date": parsed_date.isoformat(), "sloc": sloc},
            headers={"Token": f"{INVENTORY_API_KEY}"},
            timeout=15.0,
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        logger.warning("Inventory API returned %s: %s", e.response.status_code, e.response.text)
        return f"Inventory API error ({e.response.status_code}): {e.response.text}"
    except httpx.HTTPError as e:
        logger.warning("Could not reach inventory API at %s: %s", INVENTORY_API_BASE_URL, e)
        return f"Could not reach inventory API: {e}"

    # logger.debug("Inventory API request to %s succeeded", INVENTORY_API_BASE_URL)
    data = response.json()
    logger.debug("Inventory API response: %s", data)
    res = ""
    try:
        if data["code"] != 1:
            return "no data can be found"
        for record in data.get("data"):
            res += f"{day}'s {record['sloc']}'s gap is {record['absoluteGap']}. "
    # gap = data.get("data")[-1].get("absoluteGap")
    except KeyError:
        return "Inventory API error: No data"
    # if gap is None:
    #     return f"Inventory API returned no gap value for sloc={sloc} on {date}: {data}"

    # unit = data.get("unit", "")
    return res
    # return f"Inventory gap for SLOC {sloc} on {date}: 100".strip()

@tool
def check_inventory_gap_for_sku( sku: str, sloc= "WC1E", cnt :int = 365) -> str:
    """Check the inventory gap (expected vs. actual stock variance) for
    a specific storage location for a sku(required).
    Use when user ask when do we see a gap for sku SM-F976UZKAVZW or what is the gap for a sku
    User May also called it SCR, confirm with user what sloc they are looking for if no sloc is provided

    Args:
        sku: unique identifier of a material, eg. SM-F976UZKAVZW", SKU are usually between 9 and 15 chars
        sloc: sloc can only be WC1E, WR2E, WR3E, WC8E
        cnt: max days of history to look at default to 365

    """

    # if not sloc:
    #     return "Missing sloc: provide a storage location code."
    # logger.info("Inventory API request to %s succeeded", INVENTORY_API_BASE_URL)
    try:
        response = httpx.get(
            f"{INVENTORY_API_BASE_URL}/dodResearch?material={sku}&sloc={sloc}",
            # params={"date": parsed_date.isoformat(), "sloc": sloc},
            headers={"Token": f"{INVENTORY_API_KEY}"},
            timeout=15.0,
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        logger.warning("Inventory API returned %s: %s", e.response.status_code, e.response.text)
        return f"Inventory API error ({e.response.status_code}): {e.response.text}"
    except httpx.HTTPError as e:
        logger.warning("Could not reach inventory API at %s: %s", INVENTORY_API_BASE_URL, e)
        return f"Could not reach inventory API: {e}"

    # logger.debug("Inventory API request to %s succeeded", INVENTORY_API_BASE_URL)
    data = response.json()
    res = ""
    try:
        if data["code"] != 1:
            return "no data can be found"
        prev_gap = data.get("data")[0]["gap"]
        cur_cnt = 0
        for record in data.get("data"):
            if cur_cnt <= cnt:
                cur_cnt += 1
                if prev_gap != record['gap']:
                    res += (f"Found gap of {sku}'s on {record['date']} is {record['gap']}  next day's gap is {prev_gap}  ")
                    return res
            else:
                res += f"beyond {cnt} days limit "

    except KeyError:
        return "Inventory API error: No data"
    # if gap is None:
    #     return f"Inventory API returned no gap value for sloc={sloc} on {date}: {data}"

    # unit = data.get("unit", "")
    return res

@tool
def check_top_inventory_gap( day: str, sloc :str, cnt :int = 7) -> str:
    """Check the sku with largest inventory gap (expected vs. actual stock variance) for
    a specific storage location(sloc) for a given day 
    use this tool when user ask what cause the SCR gap for a sloc or which details of the gap for a sku
    or what are the gaps for a sloc in detail
    User May also called it SCR, confirm with user what sloc they are looking for

    Args:
        day: ISO date string, e.g. "2026-07-18".
        sloc: sloc can only be WC1E, WR2E, WR3E, WC8E
        cnt: number of days to pull, defaults to 7 optional, no need to confirm 

    """

    if sloc == "":
        return "Missing sloc: provide a storage location code."
    try:
        parsed_date = _date.fromisoformat(day)
        daydiff:timedelta = date.today() - parsed_date
    except ValueError:
        return f"Invalid date '{day}'. Use ISO format, e.g. 2026-07-16."
    try:
        response = httpx.get(
            f"{INVENTORY_API_BASE_URL}/scrReportGap?date={daydiff.days}&sloc={sloc}",
            headers={"Token": f"{INVENTORY_API_KEY}"},
            timeout=15.0,
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        logger.warning("Inventory API returned %s: %s", e.response.status_code, e.response.text)
        return f"Inventory API error ({e.response.status_code}): {e.response.text}"
    except httpx.HTTPError as e:
        logger.warning("Could not reach inventory API at %s: %s", INVENTORY_API_BASE_URL, e)
        return f"Could not reach inventory API: {e}"

    # logger.debug("Inventory API request to %s succeeded", INVENTORY_API_BASE_URL)
    data = response.json()
    res = ""
    try:
        if data["code"] != 1:
            return "no data can be found"
        cur_cnt = 0
        for record in data.get("data"):
            if cur_cnt <= cnt:
                cur_cnt += 1
                res += f"{record['material']} 's gap is {record['totalGap']}. "

    except KeyError:
        return "Inventory API error: No data"
    return res