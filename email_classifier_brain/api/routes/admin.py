"""
api/routes/admin.py — Admin & Log Management Endpoints
=======================================================

Protected endpoints for manual job triggers, label correction,
training data management, and update control.
"""

import hashlib
import json
import logging
import os
import uuid
from pathlib import Path

import classify
import config
import database
import imap_client
from api.security import is_trusted_ip
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request

from api.models import CandidateCategoryRequest, CorrectionRequest
from api.security import get_api_key
from job_queue import job_queue
from jobs.correction import check_corrections_job, force_check_corrections_job
from jobs.training_data import add_to_training_data, backfill_training_data_job, push_training_data_to_git
from jobs.update import append_update_history, get_retrain_status, shutdown_server

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/logs/{log_id}/correction", dependencies=[Depends(get_api_key)])
def correct_label(log_id: str, req: CorrectionRequest):
    """
    Correct the label for a specific email log.
    Updates the database and moves the email in the training data: the
    example is written to the new category's file and any copy under the
    previous category is removed (correction is a MOVE, not an append).
    """
    available_categories = classify.get_available_categories()
    if req.corrected_category not in available_categories:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid category: {req.corrected_category}. Must be one of: {', '.join(available_categories)}"
        )

    log_entry = database.get_log_by_id(log_id)
    if not log_entry:
        raise HTTPException(status_code=404, detail="Log entry not found")

    old_label = log_entry.get('corrected_category') or log_entry.get('predicted_category')

    # Write training data FIRST, then update DB.
    # If training data write fails, the DB won't be updated.
    add_to_training_data(log_entry, req.corrected_category, previous_category=old_label)
    database.update_log_correction(log_id, req.corrected_category)

    # Apply VERIFIED label in IMAP as permanent marker
    client = None
    gmail_sync_ok = True
    try:
        client = imap_client.GmailClient()
        gmail_sync_ok &= client.apply_label(log_id, req.corrected_category)
        if old_label != req.corrected_category and not client.remove_label(log_id, old_label):
            gmail_sync_ok = False
        # A human has resolved the label — clear the classifier's UNSURE flag
        # (classifier-owned only: the user's own labels are never touched)
        gmail_sync_ok &= client.remove_label(log_id, config.UNSURE_LABEL)
        gmail_sync_ok &= client.apply_label(log_id, config.VERIFICATION_LABEL)
    except Exception as e:
        gmail_sync_ok = False
        logger.error(f"Failed to update IMAP labels for {log_id}: {e}")
    finally:
        if client:
            client.disconnect()

    if not gmail_sync_ok:
        # DB-fixed-but-Gmail-stale drift: the human reads update_history.json
        append_update_history(
            "warning",
            f"Gmail sync incomplete for email {log_id}: correction to "
            f"{req.corrected_category!r} recorded (training data + DB updated) but "
            f"some label writes failed. Resync with POST /logs/{log_id}/repair-gmail-labels.",
        )

    response = {"status": "success", "message": f"Label corrected to {req.corrected_category} and added to training data."}
    if not gmail_sync_ok:
        response["gmail_sync"] = "incomplete"
        response["message"] += " (Warning: some Gmail label writes failed — see update_history.json; resync with POST /logs/{log_id}/repair-gmail-labels)"
    return response


@router.get("/logs/ambiguous", dependencies=[Depends(get_api_key)])
def get_ambiguous_logs():
    """
    Get logs that have been flagged as ambiguous (multiple trained labels found on server).
    """
    return database.get_ambiguous_logs()


def _validate_new_category(category: str) -> str:
    """
    Validate a user-proposed new category before admitting it as a training
    file. Must map to a safe single jsonl path (train.py derives the label
    from the file path; '/' creates hierarchy, path traversal is rejected).
    """
    category = (category or "").strip()
    if not category:
        raise HTTPException(status_code=400, detail="Category must not be empty")
    parts = category.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise HTTPException(status_code=400, detail=f"Invalid category path: {category!r}")
    if category in classify.get_available_categories():
        raise HTTPException(status_code=400, detail=f"Category {category} is already known to the model")
    return category


@router.get("/logs/candidate-categories", dependencies=[Depends(get_api_key)])
def get_candidate_categories():
    """
    Get logs flagged by the recheck job as carrying Gmail labels the model
    has never trained on (candidate new categories the user tagged manually).
    """
    return database.get_candidate_category_logs()


@router.post("/logs/candidate-categories/{log_id}/accept", dependencies=[Depends(get_api_key)])
def accept_candidate_category(log_id: str, req: CandidateCategoryRequest):
    """
    Consciously admit a candidate category: writes the email to
    TrainingData/<category>.jsonl (the corpus starts teaching it), records the
    correction, clears the candidate flag, and marks the email VERIFIED in
    Gmail. The new category only takes effect for classification after the
    next retrain (the served model's label set is rebuilt from the corpus).
    """
    log_entry = database.get_log_by_id(log_id)
    if not log_entry:
        raise HTTPException(status_code=404, detail="Log entry not found")

    category = _validate_new_category(req.category)

    add_to_training_data(log_entry, category)
    database.update_log_correction(log_id, category)
    database.clear_candidate_labels(log_id)

    client = None
    try:
        client = imap_client.GmailClient()
        client.apply_label(log_id, category)
        client.apply_label(log_id, config.VERIFICATION_LABEL)
    except Exception as e:
        logger.error(f"Failed to update IMAP labels for {log_id}: {e}")
    finally:
        if client:
            client.disconnect()

    return {
        "status": "success",
        "category": category,
        "message": f"Category {category} admitted to training data; classification adopts it after the next retrain.",
    }


@router.post("/logs/candidate-categories/{log_id}/discard", dependencies=[Depends(get_api_key)])
def discard_candidate_category(log_id: str):
    """
    Discard a candidate new-category flag without touching training data or
    Gmail labels. The recheck job may re-flag the email on a later pass.
    """
    cleared = database.clear_candidate_labels(log_id)
    if not cleared:
        raise HTTPException(status_code=404, detail="No candidate category flag on this log")
    return {"status": "success", "message": "Candidate category flag discarded."}


def _compute_repair_actions(live_labels: list, expected: str, known_categories: set,
                            apply_verified: bool) -> list:
    """
    Pure computation of the classifier-owned repair actions for one email's
    live labels. Shared by the apply path and the unconfirmed-refusal report
    (backslash-prefixed IMAP system artifacts are never trained categories,
    so they are naturally excluded).
    """
    actions = []
    for lbl in list(live_labels):
        if lbl in known_categories and lbl != expected:
            actions.append({"action": "remove", "label": lbl})
    if config.UNSURE_LABEL in live_labels:
        actions.append({"action": "remove_flag", "label": config.UNSURE_LABEL})
    if expected not in live_labels:
        actions.append({"action": "apply", "label": expected})
    if apply_verified and config.VERIFICATION_LABEL not in live_labels:
        actions.append({"action": "apply", "label": config.VERIFICATION_LABEL})
    return actions


def _execute_repair_action(client, log_id: str, action: dict) -> bool:
    """Execute one planned repair action; returns the write's ok flag."""
    if action["action"] == "remove":
        return client.remove_label(log_id, action["label"])
    if action["action"] == "remove_flag":
        return client.remove_label(log_id, action["label"])
    return client.apply_label(log_id, action["label"])


@router.post("/logs/{log_id}/repair-gmail-labels", dependencies=[Depends(get_api_key)])
def repair_gmail_labels(log_id: str):
    """
    USER-TRIGGERED Gmail label sync for one logged email — there is no
    automation behind it. Recomputes the classifier-owned label state from
    the database and applies it to the email:

    - removes every TRAINED category label that is not the row's current
      corrected/predicted category (the 'old label still on the message'
      drift, e.g. the 2026-10 specialkost case),
    - removes a leftover UNSURE_CLASSIFICATION flag,
    - applies the expected category and VERIFIED if absent.

    HARD RULE (2026-10): VERIFIED is only ever applied to rows whose
    corrected_category is SET — a human must have confirmed the label.
    On a model-predicted (uncorrected) row the endpoint REFUSES with a 409
    carrying the report of what it WOULD have done, so a model guess can
    never be laundered into a 'verified fact'. The repair never writes the
    database (update_log_correction is not involved) — it stays a Gmail-only,
    side-effect-bounded surface.

    The user's own labels are never touched; works for archived and
    trashed-but-not-purged mail (uid resolution covers out-of-inbox
    mailboxes). Returns a JSON action report.
    """
    log_entry = database.get_log_by_id(log_id)
    if not log_entry:
        raise HTTPException(status_code=404, detail="Log entry not found")

    corrected = log_entry.get('corrected_category')
    expected = corrected or log_entry.get('predicted_category')
    known_categories = set(classify.get_available_categories())
    if expected not in known_categories:
        raise HTTPException(
            status_code=400,
            detail=f"Row category {expected!r} is not a trained category; "
                   f"correct the row first."
        )

    confirmed = bool(corrected)

    client = None
    try:
        client = imap_client.GmailClient()
        live_labels = client.get_labels_for_emails([log_id]).get(log_id, [])
        planned = _compute_repair_actions(
            live_labels, expected, known_categories, apply_verified=confirmed
        )
        if not confirmed:
            # Model guess, not human fact: refuse, and hand back the plan the
            # OLD semantics would have executed (VERIFIED included) — that
            # laundering-in-waiting is exactly what the refusal prevents.
            refused_plan = _compute_repair_actions(
                live_labels, expected, known_categories, apply_verified=True
            )
            raise HTTPException(
                status_code=409,
                detail={
                    "status": "refused",
                    "reason": (
                        "Row is model-predicted, not human-confirmed — repair would "
                        "stamp the model's guess as VERIFIED. Confirm the label first "
                        f"via POST /logs/{log_id}/correction, then retry."
                    ),
                    "expected_category": expected,
                    "live_labels_before": live_labels,
                    "would_apply": refused_plan,
                },
            )
        actions = [
            {**a, "ok": _execute_repair_action(client, log_id, a)} for a in planned
        ]
        return {
            "status": "success",
            "expected_category": expected,
            "live_labels_before": live_labels,
            "actions": actions,
        }
    finally:
        if client:
            client.disconnect()


# ---------------------------------------------------------------------------
# Model-artifact upload/install — the STRICTEST routes in the app.
# TAILSCALE/LOCALHOST PEER ONLY: even a VALID admin key from an untrusted
# network is rejected, because these routes write code-adjacent payloads
# (a .pkl that executes on load) beyond the key wall. Trust model comes
# from the user directive 2026-10-24.
# ---------------------------------------------------------------------------

_MODEL_KIND_BY_EXT = {
    ".gguf": "llama_model",      # llama.cpp GGUF (merged LoRA model)
    ".safetensors": "st_model",  # ST model set components
    ".json": "st_model",         # config.json, label_mapping.json, MODEL_INFO.json ...
    ".pkl": "st_model",          # model_head.pkl (executes on load — hence the peer wall)
}


def _require_trusted_peer(request: Request) -> str:
    """Reject unless the socket peer is localhost or the Tailscale CGNAT
    range. Runs INSIDE the handler so a valid admin key from an untrusted
    network is still rejected (403) — the key wall alone is not enough here."""
    peer = request.client.host if request.client else None
    if not peer or not is_trusted_ip(peer):
        logger.warning("Model-artifact route refused for untrusted peer %r", peer)
        raise HTTPException(
            status_code=403,
            detail="This endpoint is only reachable from localhost/Tailscale connections.",
        )
    return peer


def _safe_artifact_name(filename: str) -> tuple:
    """basename-only + per-kind extension whitelist. Returns (name, kind)."""
    name = os.path.basename((filename or "").strip())
    if not name or name in (".", "..") or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail=f"Invalid artifact filename: {filename!r}")
    kind = _MODEL_KIND_BY_EXT.get(os.path.splitext(name)[1].lower())
    if not kind:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported artifact type (allowed: {', '.join(sorted(_MODEL_KIND_BY_EXT))})",
        )
    return name, kind


@router.post("/admin/upload-model-artifact", dependencies=[Depends(get_api_key)])
async def upload_model_artifact(
    request: Request,
    filename: str = Query(..., description="Target artifact filename (basename only)"),
    sha256_expected: str | None = Query(None, description="Optional integrity check before staging completes"),
):
    """
    Stream-upload a model artifact into the staging dir (TAILSCALE/LOCALHOST
    PEER ONLY regardless of caller key). Raw body = file bytes. sha256 is
    computed while streaming and echoed; if `sha256_expected` is given a
    mismatch aborts the staging. Nothing is installed or activated here —
    the explicit /admin/install-model-artifact call decides when to flip.
    """
    peer = _require_trusted_peer(request)
    name, kind = _safe_artifact_name(filename)

    declared = request.headers.get("content-length")
    if declared and int(declared) > config.MAX_MODEL_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Artifact exceeds upload size cap")

    os.makedirs(config.MODEL_STAGING_DIR, exist_ok=True)
    staged_path = os.path.join(config.MODEL_STAGING_DIR, name)
    tmp_path = os.path.join(config.MODEL_STAGING_DIR, f".part-{uuid.uuid4().hex}")

    sha = hashlib.sha256()
    size = 0
    try:
        async for chunk in request.stream():
            size += len(chunk)
            if size > config.MAX_MODEL_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="Artifact exceeds upload size cap")
            sha.update(chunk)
            with open(tmp_path, "ab") as f:
                f.write(chunk)
    except HTTPException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    digest = sha.hexdigest()
    if sha256_expected and digest != sha256_expected.strip().lower():
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise HTTPException(
            status_code=400,
            detail=f"sha256 mismatch: got {digest}, expected {sha256_expected}",
        )
    if size == 0:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise HTTPException(status_code=400, detail="Empty artifact")

    # atomic within staging: same-dir replace of the completed .part file
    if os.path.exists(staged_path):
        existing = hashlib.sha256()
        with open(staged_path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                existing.update(chunk)
        if existing.hexdigest() != digest:
            os.remove(tmp_path)
            raise HTTPException(
                status_code=409,
                detail=f"{name} is already staged with a different sha256; "
                       f"delete it first (install endpoint reports paths).",
            )
        os.remove(tmp_path)  # identical bytes already staged — idempotent
    else:
        os.replace(tmp_path, staged_path)

    append_update_history(
        "info",
        f"Model artifact '{name}' ({kind}) staged: {size} bytes, sha256={digest[:16]}… "
        f"from trusted peer {peer}. Not installed — "
        f"POST /admin/install-model-artifact to activate.",
    )
    return {
        "status": "staged",
        "staged_as": name,
        "kind": kind,
        "size": size,
        "sha256": digest,
        "staging_dir": config.MODEL_STAGING_DIR,
    }


@router.post("/admin/install-model-artifact", dependencies=[Depends(get_api_key)])
def install_model_artifact(request: Request, staged_name: str = Query(...,
                           description="Staged artifact filename (as reported by upload)")):
    """
    Atomically move a previously staged+verified artifact into its ACTIVE
    location (GGUF → the llama.cpp model path; ST components → model/).
    NO auto-restart, no auto-activation beyond the move — the human operator
    decides when to flip; the target service loads on its next restart.
    """
    peer = _require_trusted_peer(request)
    name, kind = _safe_artifact_name(staged_name)
    staged_path = os.path.join(config.MODEL_STAGING_DIR, name)
    if not os.path.exists(staged_path):
        raise HTTPException(status_code=404, detail=f"Nothing staged as '{staged_name}'")

    sha = hashlib.sha256()
    with open(staged_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(chunk)
    digest = sha.hexdigest()

    if kind == "llama_model":
        target = config.LLAMA_MODEL_FILE
    else:
        target = os.path.join(config.MODEL_OUTPUT_DIR, name)
    os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)

    # atomic within the same filesystem: both live under the service workdir
    os.replace(staged_path, target)

    append_update_history(
        "info",
        f"Model artifact '{name}' ({kind}) INSTALLED to {target} "
        f"(sha256={digest[:16]}…, peer {peer}). Restart the target service "
        f"(email-embedder / email-classifier) to load it.",
    )
    return {
        "status": "installed",
        "artifact": name,
        "kind": kind,
        "target": os.path.abspath(target),
        "size": os.path.getsize(target),
        "sha256": digest,
        "note": "Staged artifact installed — restart the target service to load it.",
    }


@router.post("/admin/push-training-data", dependencies=[Depends(get_api_key)])
def trigger_push_training_data():
    """
    Manually trigger pushing training data to Git.
    """
    try:
        push_ok = push_training_data_to_git()
        if not push_ok:
            raise RuntimeError("Training data git push failed (see service logs / update_history.json)")
        return {"status": "success", "message": "Training data push initiated."}
    except Exception as e:
        logger.error(f"Failed to push training data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/admin/retrain-status", dependencies=[Depends(get_api_key)])
def retrain_status():
    """
    Whether the training data has commits the deployed model was never
    trained on ('retrain worthwhile'), with both commit ids. Training is
    never triggered from here — ./retrain.sh on the trainer machine is the
    invocation path; this is the readable signal.
    """
    return get_retrain_status()


@router.post("/admin/check-corrections", dependencies=[Depends(get_api_key)])
def trigger_check_corrections():
    """
    Trigger the check corrections process for existing logs.
    """
    accepted = job_queue.enqueue("recheck", check_corrections_job, trigger="manual")
    if accepted:
        return {"status": "accepted", "message": "Check corrections queued."}
    return {"status": "already_queued", "message": "Job already queued or running."}


# WARNING: This endpoint is expensive and should ONLY be used when you have
# manually re-labelled emails in Gmail and need to pick up those corrections
# immediately (e.g. to update training data before a model retrain).
# Do NOT use this for regular periodic checks — use /admin/check-corrections instead.
@router.post("/admin/force-check-corrections", dependencies=[Depends(get_api_key)])
def trigger_force_check_corrections():
    """
    Force re-check ALL emails for label corrections, bypassing the gliding
    scale schedule. Use this after manually re-labelling emails in Gmail to
    update training data.
    """
    accepted = job_queue.enqueue("force_recheck", force_check_corrections_job, trigger="manual")
    if accepted:
        return {"status": "accepted", "message": "Force check corrections queued."}
    return {"status": "already_queued", "message": "Job already queued or running."}


@router.post("/admin/backfill-training-data", dependencies=[Depends(get_api_key)])
def trigger_backfill_training_data():
    """
    Rebuild training data files from all corrected entries in the database.
    Use this to recover training data if the training data directory was
    accidentally emptied or lost.
    """
    accepted = job_queue.enqueue("backfill", backfill_training_data_job, trigger="manual")
    if accepted:
        return {"status": "accepted", "message": "Backfill training data queued."}
    return {"status": "already_queued", "message": "Job already queued or running."}


@router.post("/admin/trigger-update", dependencies=[Depends(get_api_key)])
def trigger_update(background_tasks: BackgroundTasks):
    """
    Manually trigger the update process.
    """
    try:
        push_training_data_to_git()
        Path(".update_request").touch()
        logger.info("Update requested via API. Server will restart.")
        background_tasks.add_task(shutdown_server)
        return {"status": "update_initiated", "message": "Server will shut down and update in a few seconds."}
    except Exception as e:
        logger.error(f"Failed to initiate update: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/admin/update-errors", dependencies=[Depends(get_api_key)])
def get_update_errors():
    """
    Get the history of update attempts and errors.
    """
    history_file = Path("update_history.json")
    if not history_file.exists():
        return []

    logs = []
    try:
        with open(history_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        logs.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        logger.error(f"Error reading update history: {e}")
        raise HTTPException(status_code=500, detail="Could not read update history")

    return logs


@router.get("/admin/sweep-state", dependencies=[Depends(get_api_key)])
def get_sweep_state():
    """
    Read-sweep debug endpoint: the read-mail sweep's complete self-diagnosis —
    everything needed to answer "why is/ isn't the sweep running" remotely,
    without journal greps. Always returns 200 with best-effort data; every
    read is individually guarded so a broken diagnostic source renders as
    an explicit error string instead of masking the sweep's state.
    """
    snap = config.sweep_stats.snapshot()

    reason = snap.get("read_sweep_last_skip_reason") or "none"
    if not config.ENABLE_READ_SWEEP:
        reason = "disabled"

    # Mailbox / search-mode state (read-only peeks at the client singleton).
    # Discovery diagnostics are strings on the real client; ONLY str values
    # are surfaced so degraded/mock client shapes render as null, not as
    # repr garbage on non-str shapes.
    try:
        resolved_all_mail = getattr(imap_client.gmail_client, "_all_mail_name", None)
        last_search_mode = getattr(imap_client.gmail_client, "last_sweep_search_mode", None)
    except Exception as e:  # pragma: no cover - mock/degraded client shapes
        resolved_all_mail = None
        last_search_mode = None
        logger.warning(f"imap_client state unavailable for sweep-state: {e}")

    def _disc_value(attr_name):
        try:
            value = getattr(imap_client.gmail_client, attr_name, None)
            return value if isinstance(value, str) else None
        except Exception:
            return None

    discovery_error = _disc_value("_all_mail_discovery_error")
    list_sample = _disc_value("_all_mail_list_sample")

    # Category source diagnostics — the root-cause surface for the
    # "no trained categories" skip (vocabulary source vs classifier's source)
    categorito = {
        "backend": config.EMBEDDER_BACKEND,
        "centroids_path": str(config.CENTROIDS_PATH),
        "centroids_abspath": os.path.abspath(str(config.CENTROIDS_PATH)),
        "centroids_exists": os.path.exists(config.CENTROIDS_PATH),
        "centroids_entries": None,
        "centroids_error": None,
    }
    try:
        import embedder
        categorito["centroids_entries"] = len(
            embedder.load_centroids(config.CENTROIDS_PATH)
        )
    except Exception as e:
        categorito["centroids_error"] = f"{type(e).__name__}: {e}"

    try:
        # canonical sorted rendering for a stable remote-debug payload
        names = sorted(classify.get_available_categories())
    except Exception as e:
        names = []
        categorito["centroids_error"] = (
            f"get_available_categories raised: {type(e).__name__}: {e}"
        )
    try:
        categorito["label_mapping_entries"] = len(
            getattr(classify, "_label_mapping", {}) or {}
        )
    except Exception:
        categorito["label_mapping_entries"] = None

    counters = {
        key: value for key, value in snap.items() if key.endswith("_total")
    }

    return {
        "enabled": bool(config.ENABLE_READ_SWEEP),
        "batch_size": config.READ_SWEEP_BATCH_SIZE,
        "last_skip_reason": reason,
        "mailbox": {
            "resolved_all_mail": resolved_all_mail,
            "last_searched": snap.get("read_sweep_mailbox_last") or None,
            "last_search_mode": last_search_mode,
            "discovery_error": discovery_error,
            "list_sample": list_sample,
        },
        "last_cycle": {
            "at": snap.get("read_sweep_last_cycle_at") or None,
            "skip_reason": reason,
            "search_mode": last_search_mode,
        },
        "counters": counters,
        "trained_categories": {
            "count": len(names),
            "names": names,
        },
        "category_source": categorito,
    }
