"""
retry.py — Generic retry utility with exponential backoff.
"""

import logging
import time
from typing import Callable, Optional, Tuple, Type

logger = logging.getLogger(__name__)


def with_retry(
    fn: Callable,
    *args,
    retries: int = 3,
    backoff: float = 1.0,
    exceptions: Tuple[Type[BaseException], ...] = (Exception,),
    on_retry: Optional[Callable] = None,
    **kwargs,
):
    """
    Call ``fn(*args, **kwargs)``, retrying on *exceptions* with exponential backoff.

    :param fn: callable to invoke
    :param retries: total number of attempts (default 3)
    :param backoff: base wait in seconds; doubles each attempt (default 1.0)
    :param exceptions: exception types that trigger a retry
    :param on_retry: optional callable(exc, attempt) invoked before each sleep
    """
    for attempt in range(1, retries + 1):
        try:
            return fn(*args, **kwargs)
        except exceptions as exc:
            if attempt >= retries:
                raise
            if on_retry is not None:
                on_retry(exc, attempt)
            wait = backoff * (2 ** (attempt - 1))
            logger.warning(
                "Attempt %d/%d failed (%s: %s). Retrying in %.1fs...",
                attempt, retries, type(exc).__name__, exc, wait,
            )
            time.sleep(wait)
