import datetime
import os
import pytest
import sys
import threading
import time

# Add the brain directory to sys.path to resolve imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../email_classifier_brain')))

from job_queue import JobQueue

@pytest.fixture
def queue():
    jq = JobQueue()
    yield jq
    jq.shutdown(timeout=2.0)

def test_enqueue_and_execute(queue):
    result = []
    def my_job():
        result.append(1)
        
    accepted = queue.enqueue("add_one", my_job)
    assert accepted is True
    
    # Wait for completion
    queue._drain()
    
    assert result == [1]

def test_dedup_while_queued(queue):
    result = []
    
    # Pause the worker so we can queue multiple
    queue._stop.set()
    queue._has_work.set()
    queue._worker.join()
    
    def my_job():
        result.append(1)
        
    acc1 = queue.enqueue("add_one", my_job)
    acc2 = queue.enqueue("add_one", my_job)
    
    assert acc1 is True
    assert acc2 is False
    assert len(queue._queue) == 1

def test_dedup_while_running(queue):
    result = []
    evt_started = threading.Event()
    evt_finish = threading.Event()
    
    def my_job():
        evt_started.set()
        evt_finish.wait()
        result.append(1)
        
    # First enqueue gets accepted
    acc1 = queue.enqueue("slow", my_job)
    assert acc1 is True
    
    # Wait until it's actually running
    evt_started.wait()
    assert queue._running == "slow"
    
    # Second enqueue while running gets rejected
    acc2 = queue.enqueue("slow", my_job)
    assert acc2 is False
    
    # Let it finish
    evt_finish.set()
    # Wait for completion
    time.sleep(0.1)
    queue._drain()
    
    assert result == [1]

def test_re_enqueue_after_completion(queue):
    result = []
    
    def my_job():
        result.append(1)
        
    acc1 = queue.enqueue("fast", my_job)
    assert acc1 is True
    
    # Drain (synchronous execution)
    queue._drain()
    assert result == [1]
    
    # Now that it's done, we can enqueue again
    acc2 = queue.enqueue("fast", my_job)
    assert acc2 is True
    
    queue._drain()
    assert result == [1, 1]

def test_sequential_execution(queue):
    events = []
    lock = threading.Lock()
    job2_finished = threading.Event()

    def job1():
        with lock:
            events.append("j1_start")
        time.sleep(0.05)
        with lock:
            events.append("j1_end")
        
    def job2():
        with lock:
            events.append("j2_start")
        time.sleep(0.05)
        with lock:
            events.append("j2_end")
        job2_finished.set()
        
    queue.enqueue("job1", job1)
    queue.enqueue("job2", job2)
    
    # Wait for the second job to finish, which implies the first also finished.
    assert job2_finished.wait(timeout=2), "Jobs did not complete in time."
    
    assert events == ["j1_start", "j1_end", "j2_start", "j2_end"]

def test_shutdown(queue):
    # Just verify shutdown method works without hanging
    queue.shutdown(timeout=0.5)
    assert not queue._worker.is_alive()

def test_status_idle(queue):
    """Status with nothing queued/running returns empty state."""
    s = queue.status()
    assert s["running"] is None
    assert s["queued"] == []

def test_status_queued(queue):
    """Enqueued-but-not-yet-started jobs appear in queued list."""
    # Stop the worker so jobs don't get picked up immediately
    queue._stop.set()
    queue._has_work.set()
    queue._worker.join()

    queue.enqueue("job_a", lambda: None)
    queue.enqueue("job_b", lambda: None)

    s = queue.status()
    assert s["running"] is None
    assert len(s["queued"]) == 2
    assert s["queued"][0]["name"] == "job_a"
    assert s["queued"][1]["name"] == "job_b"
    for entry in s["queued"]:
        assert entry["enqueued_at"] is not None
        assert entry["started_at"] is None
        datetime.datetime.fromisoformat(entry["enqueued_at"])

def test_status_running(queue):
    """The currently running job appears in status['running'] with started_at set."""
    evt_started = threading.Event()
    evt_finish = threading.Event()

    def slow_job():
        evt_started.set()
        evt_finish.wait()

    queue.enqueue("slow", slow_job)
    evt_started.wait(timeout=2)

    s = queue.status()
    assert s["running"] is not None
    assert s["running"]["name"] == "slow"
    assert s["running"]["enqueued_at"] is not None
    assert s["running"]["started_at"] is not None
    datetime.datetime.fromisoformat(s["running"]["started_at"])
    assert s["queued"] == []

    evt_finish.set()
    time.sleep(0.05)
    queue._drain()

    s2 = queue.status()
    assert s2["running"] is None
