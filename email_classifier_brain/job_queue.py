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
import logging
import threading
from typing import Any, Callable, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class JobQueue:
    """Thread-safe, deduplicated, sequential job queue."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # Ordered dict so we process in enqueue order
        self._queue: collections.OrderedDict[str, Tuple[Callable, tuple, dict]] = (
            collections.OrderedDict()
        )
        self._running: Optional[str] = None
        self._stop = threading.Event()
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
            self._queue[name] = (fn, args, kwargs)
            logger.info(f"Job '{name}' enqueued (queue depth: {len(self._queue)}).")
            self._has_work.set()
            return True

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

            if self._stop.is_set():
                # Drain remaining work before exiting
                self._drain()
                break

            self._drain()

    def _drain(self) -> None:
        """Process all currently queued jobs."""
        while True:
            with self._lock:
                if not self._queue:
                    self._has_work.clear()
                    self._running = None
                    return
                name, (fn, args, kwargs) = self._queue.popitem(last=False)
                self._running = name

            # Execute outside the lock so new items can be enqueued
            logger.info(f"Job '{name}' starting.")
            try:
                fn(*args, **kwargs)
            except Exception:
                logger.exception(f"Job '{name}' failed with exception.")
            finally:
                logger.info(f"Job '{name}' finished.")
                with self._lock:
                    self._running = None
