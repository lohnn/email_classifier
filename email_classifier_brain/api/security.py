"""
api/security.py — API Key Authentication
=========================================

Provides the API key dependency used to protect admin endpoints.
Requests from localhost or Tailscale CGNAT range bypass the key check.

Two key tiers:
- ADMIN_API_KEY - full access: reads AND mutations (unchanged).
- READ_API_KEY (optional) - read-only: accepted for GET/HEAD on key-gated
  endpoints; mutating routes still require the admin key or the
  trusted-network bypass. Unset/empty READ_API_KEY changes nothing.
Keys are never logged.
"""

import logging
import os
from ipaddress import ip_address, ip_network

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader

logger = logging.getLogger(__name__)

api_key_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)

TRUSTED_NETWORKS = [
    ip_network("127.0.0.0/8"),
    ip_network("::1/128"),
    ip_network("100.64.0.0/10"),  # Tailscale CGNAT range
]

# Methods that only ever read state. Everything else (POST today) is a
# mutation and never accepts the read-only key.
SAFE_METHODS = {"GET", "HEAD"}


def is_trusted_ip(ip: str) -> bool:
    """Return True if the IP belongs to localhost or Tailscale."""
    try:
        addr = ip_address(ip)
        return any(addr in net for net in TRUSTED_NETWORKS)
    except ValueError:
        return False


def _is_valid_read_key(api_key: str | None) -> bool:
    """Check if the provided key matches READ_API_KEY (read-only tier)."""
    if api_key is None:
        return False
    expected_read_key = os.getenv("READ_API_KEY")
    if not expected_read_key:
        return False
    return api_key == expected_read_key


def is_valid_admin_key(api_key: str | None) -> bool:
    """Check if the provided API key matches ADMIN_API_KEY from the environment."""
    expected_key = os.getenv("ADMIN_API_KEY")
    if not expected_key:
        return False
    return api_key == expected_key


def get_api_key(
    request: Request,
    api_key: str = Security(api_key_scheme),
):
    """
    Validates the API key against ADMIN_API_KEY in the environment.
    Requests from trusted networks (localhost, Tailscale) skip the check.
    A valid READ_API_KEY grants GET/HEAD only; mutating methods still
    require the admin key (or the trusted-network bypass).
    If neither key applies and ADMIN_API_KEY is not set, access is denied
    (500) for untrusted clients, exactly as before.
    """
    client_ip = request.client.host if request.client else None
    if client_ip and is_trusted_ip(client_ip):
        return api_key or "trusted-network"

    # Read-only tier: a valid READ_API_KEY grants GET/HEAD on key-gated
    # endpoints. Mutating methods fall through to the admin check (a read
    # key is not an admin key) and are rejected there.
    if request.method in SAFE_METHODS and _is_valid_read_key(api_key):
        return api_key

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
