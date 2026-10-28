"""
retrain_store.py — Retrain-Event Status Mirror (JSONL store)
============================================================

The queryable mirror of the retraining orchestrator's events (design:
docs/RETRAIN-ORCHESTRATOR-DESIGN.md §6). The Mac retrain orchestrator is
the SOURCE OF TRUTH; this store is the mirror that persists when the Mac
is offline and that the operator, deploy automation, or a fresh session
can query over HTTP:

    POST /admin/retrain-events   record one event (idempotent, see below)
    GET  /admin/retrain-events   {latest, history}

Storage (decided with the operator 2026-10-28, design §11 Q1):
`retrain_events.json` in the service workdir — one event per line (JSONL,
same file discipline as update_history.json), keeping the ~50 newest
entries. Every recorded event ALSO appends one human-readable line to
update_history.json via jobs.update.append_update_history: the human reads
that file; this store is the queryable source. The mirror records — it
never decides; run state lives in the orchestrator's own ledger.

Idempotency (at-least-once delivery → exactly-once recording):
the uniqueness key is the (run_id, event) PAIR. A run legitimately emits
several events (run_started, run_finished, …) under one run_id; a retried
POST of the same (run_id, event) with any payload — including a different
`at` — is answered with the already-recorded event and stores nothing.
Dedup covers the retained window (the ~50 newest events); deeper history
is not kept anyway.
"""

import json
import logging
import os
import threading

import jobs.update

logger = logging.getLogger(__name__)

# Same workdir-relative convention as jobs/update.py HISTORY_FILE: the
# service workdir (repo checkout on the Pi) is where run_service.sh writes
# update_history.json, and where this mirror lives.
RETRAIN_EVENTS_FILE = "retrain_events.json"
MAX_RETRAIN_EVENTS = 50

_store_lock = threading.Lock()


def load_retrain_events() -> list:
    """
    Read every retained event, in file order (oldest → newest). Corrupt or
    blank lines are skipped (a torn line from a crash must not take the
    whole history down), so the returned list may be shorter than the file
    has lines.
    """
    events = []
    path = RETRAIN_EVENTS_FILE
    if not os.path.exists(path):
        return events
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("Skipping unparseable line in %s", path)
    except OSError as e:
        logger.error("Could not read %s: %s", path, e)
    return events


def record_retrain_event(event: dict) -> dict:
    """
    Append one already-validated event dict to the JSONL store — unless its
    (run_id, event) pair is already present in the retained window, in which
    case the FIRST record stands and nothing is written (duplicate).

    Returns {"recorded": bool, "duplicate": bool, "event": <stored event>}:
    the stored event is the first recording in the duplicate case, the new
    event otherwise. A recorded event additionally gets its human-readable
    line in update_history.json; duplicates must not — the human file stays
    exactly one line per recorded event.
    """
    with _store_lock:
        if not os.path.exists(RETRAIN_EVENTS_FILE):
            _rewrite_file([event])
        else:
            events = load_retrain_events()
            for existing in events:
                if (existing.get("run_id") == event["run_id"]
                        and existing.get("event") == event["event"]):
                    logger.info(
                        "Retrain event duplicate ignored: %s / %s (already recorded at %s)",
                        event["run_id"], event["event"], existing.get("at"),
                    )
                    return {"recorded": False, "duplicate": True, "event": existing}
            events.append(event)
            _rewrite_file(events)

    jobs.update.append_update_history("info", _human_line(event))
    return {"recorded": True, "duplicate": False, "event": event}


def _rewrite_file(events: list) -> None:
    """Atomically persist the event list oldest→newest, keeping the last
    MAX_RETRAIN_EVENTS entries (same .tmp + os.replace pattern as
    jobs.update.append_update_history)."""
    events = events[-MAX_RETRAIN_EVENTS:]
    tmp_path = RETRAIN_EVENTS_FILE + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            for ev in events:
                f.write(json.dumps(ev, ensure_ascii=False) + "\n")
        os.replace(tmp_path, RETRAIN_EVENTS_FILE)
    except OSError as e:
        logger.error("Could not write %s: %s", RETRAIN_EVENTS_FILE, e)
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass


def _human_line(event: dict) -> str:
    """
    One compact, operator-readable sentence describing a recorded event —
    the update_history.json rendering of the mirror entry.
    """
    run_id = event.get("run_id") or "?"
    name = event.get("event")
    error = (event.get("error") or "").strip()
    if name == "run_started":
        text = f"Retrain run {run_id} started on the orchestrator Mac."
    elif name == "run_finished":
        text = f"Retrain run {run_id} finished."
    elif name == "run_failed":
        text = f"Retrain run {run_id} FAILED: {error or 'unspecified error'}."
    elif name == "skipped":
        text = f"Retrain run {run_id} skipped: {error or 'unknown reason'}."
    elif name == "pr_opened":
        text = f"Vocabulary PR opened for retrain run {run_id}."
    elif name == "deploy_verified":
        text = f"Deploy of retrain run {run_id} verified on the classifier."
    else:  # future event kind — never lose it from the human log
        text = f"Retrain event {name or '?'} for run {run_id}."

    bits = []
    if eval_data := event.get("eval"):
        acc = eval_data.get("accuracy")
        if isinstance(acc, (int, float)):
            bits.append(f"accuracy {acc:.3f}")
    data_sha = event.get("data_sha")
    if data_sha:
        bits.append(f"data {data_sha[:12]}")
    durations = event.get("durations")
    if isinstance(durations, dict) and durations:
        bits.append(f"{sum(durations.values()):.0f}s total")
    if bits:
        text = f"{text.rstrip('.')} ({', '.join(bits)})."
    return text
