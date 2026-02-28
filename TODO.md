# TODO

Potential improvements to the server, roughly ordered by impact.

---

## Observability & Operations

- [ ] **Real health check** (`GET /health`) Currently just returns
      `{"status": "ok"}` unconditionally. Should verify DB connectivity, model
      loaded state, and optionally IMAP reachability — useful behind a load
      balancer or uptime monitor.

- [ ] **Job status endpoint** (`GET /jobs/status`) After hitting `/run` or
      `/reclassify` there is no feedback beyond "accepted". An endpoint
      returning what is currently running/queued (job name, enqueued time,
      started time) would let the UI show real progress.

- [ ] **Job cancellation** (`POST /jobs/cancel`) Currently, if a heavy job (like
      `reclassify_job`) gets stuck or takes hours, it strictly blocks the
      single-worker background thread. Adding a way to clear the queue or cancel
      the currently running job would be a huge operational improvement.

- [ ] **Job run history** Store per-run metadata (last run time, duration,
      emails processed, error count) in the DB or a sidecar file. Currently only
      queryable by grepping logs.

---

## Data & API

- [ ] **Pagination on list endpoints** `/notifications`, `/logs/ambiguous`, and
      `/stats` return unbounded result sets. Add `page`/`limit` query params
      before the DB grows large enough to cause latency issues.

- [ ] **Stats reflect corrections** `get_stats` counts by `predicted_category`,
      not `corrected_category`. After many user corrections the dashboard
      becomes misleading. At minimum, surface corrected counts alongside
      predicted counts.

- [ ] **Log browsing endpoint** (`GET /logs`) No way to list or search the full
      email history via the API. A filterable, paginated endpoint (by category,
      date range, correction status) would unlock a history view in the UI.

- [ ] **View single email detail endpoint** (`GET /logs/{log_id}`) The API
      supports updating a specific log's correction via its ID, but there is no
      endpoint to just fetch a specific email's details. Useful for opening an
      email "detail view" in the dashboard.

- [ ] **Blocklist / Ignore List** No mechanism exists to completely ignore
      persistent noisy senders or domains. Adding an ignore list table to the DB
      and exposing an endpoint to manage it would save processing power and DB
      space from irrelevant daily marketing emails.

- [ ] **Model info endpoint** (`GET /model/info`) `train.py` already writes
      `MODEL_INFO.json` (training timestamp, base model, categories, sample
      counts, git provenance) but it is never exposed. Surfacing it through the
      API would let the UI show when the model was last trained.

---

## Resilience

- [ ] **Retry logic on IMAP and Git** All IMAP, Git, and DB failures log the
      error and move on — no retries. A transient IMAP disconnect silently drops
      the entire classification batch. Even a single retry with a short backoff
      would meaningfully improve reliability.

- [ ] **IMAP connection reuse** A fresh SSL connection is opened and closed for
      every job invocation. With a 5-minute classification interval this is
      frequent churn. Reusing a persistent connection (with reconnect on
      failure) would be faster and less wasteful.

- [ ] **Graceful shutdown on auto-update** The auto-update job sends `SIGTERM`
      to itself while a classification job may be in-flight. The job queue
      should drain before the process exits to avoid losing in-progress work.

---

## Security

- [ ] **Rate limiting** Any valid API key can hammer endpoints like `/run` or
      `/reclassify` indefinitely. A simple per-key rate limit would prevent
      accidental or intentional overload.

- [ ] **Read vs. write API key scopes** A single key controls everything from
      reading stats to triggering restarts. Splitting into read-only and admin
      scopes would let the UI use a key that cannot trigger destructive
      operations.

---

## Testing Gaps

The following areas have no test coverage:

- [ ] `check_corrections_job` and the `_resolve_correction` state machine
- [ ] `reclassify_job`
- [ ] `force_check_corrections_job` / `backfill_training_data_job`
- [ ] `train.py` (training pipeline end-to-end)
- [ ] `add_to_training_data` deduplication logic
- [ ] `get_candidate_logs_for_recheck` (gliding-scale SQL query)
- [ ] `GmailClient.apply_label`, `remove_label`, `get_labels_for_emails`,
      `scan_labeled_emails`
- [ ] DB schema migration path
- [ ] Stats time-range filtering
