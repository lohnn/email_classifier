"""
api/security.py — API Key Authentication
==========================================

Provides the API key dependency used to protect admin endpoints.
Requests from localhost or Tailscale CGNAT range bypass the key check.
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


def is_trusted_ip(ip: str) -> bool:
    """Return True if the IP belongs to localhost or Tailscale."""
    try:
        addr = ip_address(ip)
        return any(addr in net for net in TRUSTED_NETWORKS)
    except ValueError:
        return False


def get_api_key(
    request: Request,
    api_key: str = Security(api_key_scheme),
):
    """
    Validates the API key against ADMIN_API_KEY in the environment.
    Requests from trusted networks (localhost, Tailscale) skip the check.
    If ADMIN_API_KEY is not set, access is denied (500) for untrusted clients.
    """
    client_ip = request.client.host if request.client else None
    if client_ip and is_trusted_ip(client_ip):
        return api_key or "trusted-network"

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
