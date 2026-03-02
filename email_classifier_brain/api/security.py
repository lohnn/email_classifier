"""
api/security.py — API Key Authentication
==========================================

Provides the API key dependency used to protect admin endpoints.
"""

import logging
import os

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

logger = logging.getLogger(__name__)

api_key_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_api_key(api_key: str = Security(api_key_scheme)):
    """
    Validates the API key against ADMIN_API_KEY in the environment.
    If ADMIN_API_KEY is not set, access is denied (500).
    """
    expected_key = os.getenv("ADMIN_API_KEY")
    if not expected_key:
        logger.error("ADMIN_API_KEY not set in environment. Blocking admin access.")
        raise HTTPException(
            status_code=500,
            detail="Server configuration error: ADMIN_API_KEY not set"
        )

    if api_key != expected_key:
        raise HTTPException(
            status_code=403,
            detail="Could not validate credentials"
        )
    return api_key
