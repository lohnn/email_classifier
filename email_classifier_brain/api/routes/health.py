"""
api/routes/health.py — Health & Stats Endpoints
=================================================

Health check endpoint and classification statistics.
"""

import datetime
import logging
from typing import Optional

import classify
import database
import imap_client
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Security
from fastapi.responses import JSONResponse

from api.models import StatsResponse
from api.security import api_key_scheme, get_api_key, is_trusted_ip, is_valid_admin_key

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
def health_check(
    request: Request,
    check_imap: bool = Query(False, description="Also verify IMAP connectivity (requires X-API-Key)"),
    api_key: str = Security(api_key_scheme),
):
    """
    Health check endpoint.

    Always verifies DB connectivity and model load state.
    Pass ?check_imap=true to also probe IMAP reachability (requires X-API-Key header).

    Returns HTTP 200 when healthy or degraded (optional IMAP failure only).
    Returns HTTP 503 when a critical component (DB or model) is unavailable.
    """
    # IMAP check is gated behind authentication to prevent unauthenticated DoS
    if check_imap:
        client_ip = request.client.host if request.client else None
        trusted = client_ip and is_trusted_ip(client_ip)
        if not trusted and not is_valid_admin_key(api_key):
            raise HTTPException(status_code=401, detail="X-API-Key required to use check_imap")

    checks: dict = {}
    critical_ok = True
    degraded = False

    # --- DB connectivity ---
    try:
        conn = database.get_db_connection()
        conn.execute("SELECT 1")
        conn.close()
        checks["database"] = {"status": "ok"}
    except Exception as e:
        logger.error(f"Health check DB error: {e}")
        checks["database"] = {"status": "error", "detail": "Database connectivity error"}
        critical_ok = False

    # --- Model loaded state ---
    if classify._model is not None:
        checks["model"] = {"status": "ok"}
    else:
        checks["model"] = {"status": "not_loaded"}
        critical_ok = False

    # --- IMAP reachability (optional, authenticated) ---
    if check_imap:
        client = None
        try:
            client = imap_client.GmailClient()
            client.connect()
            checks["imap"] = {"status": "ok"}
        except ValueError:
            checks["imap"] = {"status": "not_configured"}
        except Exception as e:
            logger.error(f"Health check IMAP error: {e}")
            checks["imap"] = {"status": "error", "detail": "IMAP connectivity error"}
            degraded = True
        finally:
            if client:
                client.disconnect()

    if not critical_ok:
        overall = "error"
        http_status = 503
    elif degraded:
        overall = "degraded"
        http_status = 200
    else:
        overall = "ok"
        http_status = 200

    return JSONResponse(status_code=http_status, content={"status": overall, "checks": checks})


@router.get("/stats", response_model=StatsResponse, dependencies=[Depends(get_api_key)])
def get_stats(
    start_time: Optional[datetime.datetime] = None,
    end_time: Optional[datetime.datetime] = None
):
    """
    Get classification statistics (counts per category).
    Optionally filter by start_time and end_time (ISO format).
    """
    stats = database.get_stats(start_time, end_time)
    return {"stats": stats}
