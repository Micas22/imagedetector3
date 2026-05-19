"""
queue_manager.py — Process-wide scan job queue.

Ensures only one scan (webapp or API) runs at a time.  Subsequent requests
are queued and assigned a position.  Callers can poll their position or
block until it is their turn.
"""

from __future__ import annotations

import threading
import uuid
from collections import OrderedDict
from typing import Optional


class ScanQueue:
    """Serialises scan jobs so only one runs at a time.

    Usage::

        job_id = scan_queue.enqueue()      # grab a ticket
        scan_queue.wait_for_turn(job_id)    # blocks until you're at the front
        try:
            ... run the actual scan ...
        finally:
            scan_queue.mark_done(job_id)    # release so the next job can start
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        # Ordered mapping  job_id -> True/False  (True = currently running).
        self._jobs: OrderedDict[str, bool] = OrderedDict()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enqueue(self) -> str:
        """Add a new job to the back of the queue.  Returns a unique job ID."""
        job_id = uuid.uuid4().hex[:12]
        with self._condition:
            self._jobs[job_id] = False
            # If this is the only job, it can start immediately.
            if len(self._jobs) == 1:
                self._jobs[job_id] = True
            self._condition.notify_all()
        return job_id

    def get_position(self, job_id: str) -> int:
        """Return 0-based queue position.  0 means *currently running*.

        Returns -1 if *job_id* is unknown (already finished or cancelled).
        """
        with self._lock:
            for idx, jid in enumerate(self._jobs):
                if jid == job_id:
                    return idx
        return -1

    def get_queue_length(self) -> int:
        """Total number of jobs (running + waiting)."""
        with self._lock:
            return len(self._jobs)

    def wait_for_turn(self, job_id: str, poll_interval: float = 0.5) -> None:
        """Block until *job_id* is at position 0 and marked as running."""
        with self._condition:
            while True:
                # Job already removed (cancelled / unknown) — bail out.
                if job_id not in self._jobs:
                    return
                first_id = next(iter(self._jobs))
                if first_id == job_id:
                    self._jobs[job_id] = True
                    self._condition.notify_all()
                    return
                # Not our turn yet — wait and re-check.
                self._condition.wait(timeout=poll_interval)

    def mark_done(self, job_id: str) -> None:
        """Signal that *job_id* has finished.  Promotes the next job."""
        with self._condition:
            self._jobs.pop(job_id, None)
            # Promote the new head of the queue (if any).
            if self._jobs:
                first_id = next(iter(self._jobs))
                self._jobs[first_id] = True
            self._condition.notify_all()

    def cancel(self, job_id: str) -> bool:
        """Remove a waiting job from the queue.  Returns True if it was found.

        If the cancelled job was at the front (running), the next job is
        promoted automatically — identical to ``mark_done``.
        """
        with self._condition:
            if job_id not in self._jobs:
                return False
            was_running = self._jobs.pop(job_id)
            if was_running and self._jobs:
                first_id = next(iter(self._jobs))
                self._jobs[first_id] = True
            self._condition.notify_all()
            return True

    def is_running(self, job_id: str) -> bool:
        """Return True if *job_id* is at the front and actively running."""
        with self._lock:
            return self._jobs.get(job_id, False) is True


# ---------------------------------------------------------------------------
# Global singleton — shared by the Streamlit webapp and the FastAPI API.
# Because Streamlit re-imports modules on every script run but keeps the
# same Python process, a module-level instance is the simplest way to
# share state across sessions and endpoints.
# ---------------------------------------------------------------------------

scan_queue = ScanQueue()
