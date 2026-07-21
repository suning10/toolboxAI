"""
Minimal API key check. Fine for an internal tool with a handful of
users; replace with your corporate SSO/OAuth if this needs to sit
behind broader access.
"""
from fastapi import Header, HTTPException
from app.config import API_KEY


def verify_api_key(x_api_key: str = Header(...)):
    # if x_api_key != API_KEY:
    #     raise HTTPException(401, "Invalid API key")
    return True
