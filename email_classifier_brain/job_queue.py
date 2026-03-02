"""
job_queue.py — Deduplicated Sequential Job Queue
=================================================

A single-worker queue that processes jobs one at a time.  Each job has a
unique *name* (e.g. "classification", "recheck").  The same name cannot
appear in the queue more than once, and a name that is currently being
executed is also considered "present" — so it can only be re-enqueued
after the current run finishes.

This replaces the old shared ``threading.Lock`` approach which silently
skipped jobs when the lock was held by another job.
"""

import collections
import datetime
import logging
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class JobQueue:
    """Thread-safe, deduplicated, sequential job queue."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # Ordered dict so we process in enqueue order
        # Values: (fn, args, kwargs, enqueued_at)
        self._queue: collections.OrderedDict[str, Tuple[Callable, tuple, dict, datetime.datetime]] = (
            collections.OrderedDict()
        )
        self._running: Optional[str] = None
        self._running_enqueued_at: Optional[datetime.datetime] = None
        self._running_started_at: Optional[datetime.datetime] = None
        self._stop = threading.Event()
        self._cancel = threading.Event()
        self._has_work = threading.Event()
        self._worker = threading.Thread(target=self._run, daemon=True, name="job-queue-worker")
        self._worker.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enqueue(self, name: str, fn: Callable, *args: Any, **kwargs: Any) -> bool:
        """Add a job to the queue.

        Returns ``True`` if the job was accepted, ``False`` if *name* is
        already queued or currently running.
        """
        with self._lock:
            if name == self._running or name in self._queue:
                logger.info(f"Job '{name}' already queued or running — skipping.")
                return False
            self._queue[name] = (fn, args, kwargs, datetime.datetime.now(datetime.timezone.utc))
            logger.info(f"Job '{name}' enqueued (queue depth: {len(self._queue)}).")
            self._has_work.set()
            return True

    def status(self) -> Dict[str, Any]:
        """Return a snapshot of the current queue state.

        Returns a dict with:
          - ``running``: info about the currently executing job, or ``None``
          - ``queued``: list of waiting jobs in order
        """
        with self._lock:
            running = None
            if self._running is not None:
                running = {
                    "name": self._running,
                    "enqueued_at": self._running_enqueued_at.isoformat() if self._running_enqueued_at else None,
                    "started_at": self._running_started_at.isoformat() if self._running_started_at else None,
                }
            queued = [
                {
                    "name": name,
                    "enqueued_at": enqueued_at.isoformat(),
                    "started_at": None,
                }
                for name, (_fn, _args, _kwargs, enqueued_at) in self._queue.items()
            ]
            return {"running": running, "queued": queued}

    def cancel(self) -> Dict[str, Any]:
        """Cancel the currently running job and clear all pending jobs.

        Sets a cancellation flag that long-running jobs should check via
        ``is_cancelled()``. The running job will not be forcefully stopped —
        it must cooperate by checking ``is_cancelled()`` and exiting early.

        Returns a dict with:
          - ``cancelled_job``: name of the job that was signalled, or ``None``
          - ``cleared_queue``: list of job names that were removed from the queue
        """
        with self._lock:
            cleared = list(self._queue.keys())
            self._queue.clear()
            running = self._running
            if running:
                self._cancel.set()
        logger.info(f"Cancel requested. Running job: {running!r}. Cleared queue: {cleared}.")
        return {"cancelled_job": running, "cleared_queue": cleared}

    def is_cancelled(self) -> bool:
        """Return True if cancellation has been requested for the current job."""
        return self._cancel.is_set()

    def shutdown(self, timeout: float = 60) -> None:
        """Signal the worker to stop and wait for it to finish."""
        logger.info("JobQueue shutdown requested.")
        self._stop.set()
        self._has_work.set()  # wake worker so it can see the stop flag
        self._worker.join(timeout=timeout)
        if self._worker.is_alive():
            logger.warning("JobQueue worker did not stop within timeout.")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Worker loop — runs on the background thread."""
        while not self._stop.is_set():
            # Wait for work (or stop signal)
            self._has_work.wait()

            # Drain any work that has been queued. If the stop flag is set,
            # the loop will terminate after this drain.
            self._drain()

    def _reset_running_status(self) -> None:
        """Clear the running job's state. Must be called while holding ``_lock``."""
        self._running = None
        self._running_enqueued_at = None
        self._running_started_at = None

    def _drain(self) -> None:
        """Process all queued jobs. Protected internally by the same lock."""
        while True:
            with self._lock:
                if not self._queue:
                    self._has_work.clear()
                    self._reset_running_status()
                    return
                # Pop the oldest job
                name, (fn, args, kwargs, enqueued_at) = self._queue.popitem(last=False)
                self._running = name
                self._running_enqueued_at = enqueued_at
                self._running_started_at = datetime.datetime.now(datetime.timezone.utc)

            try:
                fn(*args, **kwargs)
            except Exception:
                logger.exception(f"Job '{name}' failed with an exception.")
            finally:
                self._cancel.clear()  # Reset cancellation flag for the next job
                with self._lock:
                    self._reset_running_status()
