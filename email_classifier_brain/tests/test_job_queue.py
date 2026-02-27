import pytest
import threading
import time
import os
import sys

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
    
    def job1():
        events.append("j1_start")
        time.sleep(0.1)
        events.append("j1_end")
        
    def job2():
        events.append("j2_start")
        time.sleep(0.1)
        events.append("j2_end")
        
    # Since worker is running, might start job1 immediately.
    # Stop worker to enqueue both sequentially.
    queue._stop.set()
    queue._has_work.set()
    queue._worker.join()
    
    # Clear stop flags and create a new worker
    queue._stop.clear()
    
    # Enqueue both
    queue.enqueue("job1", job1)
    queue.enqueue("job2", job2)
    
    # Process sequentially inline
    queue._drain()
    
    assert events == ["j1_start", "j1_end", "j2_start", "j2_end"]

def test_shutdown(queue):
    # Just verify shutdown method works without hanging
    queue.shutdown(timeout=0.5)
    assert not queue._worker.is_alive()
