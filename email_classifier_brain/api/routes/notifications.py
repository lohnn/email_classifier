"""
api/routes/notifications.py — Notification Endpoints
=====================================================

Endpoints for reading, acknowledging, and popping email
classification notifications.
"""

import datetime
import logging
from typing import List

import database
from fastapi import APIRouter, Depends

from api.models import AckRequest, Notification
from api.security import get_api_key

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/notifications")


@router.get("", response_model=List[Notification], dependencies=[Depends(get_api_key)])
def get_notifications():
    """
    Get all unread notifications.
    """
    return database.get_unread_notifications()


@router.post("/ack", dependencies=[Depends(get_api_key)])
def ack_notifications(req: AckRequest):
    """
    Acknowledge notifications (mark as read).
    If `ids` is provided, marks those specific IDs.
    If `ids` is empty or null, marks ALL unread notifications.
    """
    database.ack_notifications(req.ids)
    return {"status": "success"}


@router.post("/pop", response_model=List[Notification], dependencies=[Depends(get_api_key)])
def pop_notifications():
    """
    Get all unread notifications AND mark them as read immediately.
    Useful for one-time fetch-and-ack clients.
    """
    return database.pop_unread_notifications()


@router.get("/read", response_model=List[Notification], dependencies=[Depends(get_api_key)])
def get_read_notifications(
    start_time: datetime.datetime,
    end_time: datetime.datetime
):
    """
    Get already read notifications within a time range.
    Start and end times are required.
    """
    return database.get_read_notifications(start_time, end_time)
