# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Structure

This is a monorepo with two subprojects:

- **`email_classifier_brain/`** — Python/FastAPI microservice for email classification (ML backend)
- **`email_classifier_ui/`** — Flutter/Riverpod cross-platform companion app

---

## Brain (Python/FastAPI)

### Commands

```bash
# From email_classifier_brain/
source venv/bin/activate

# Run the service
python main.py  # or: uvicorn main:app --host 0.0.0.0 --port 8008

# Train the model
python train.py

# Run all tests
pytest tests/

# Run a single test file
pytest tests/test_app.py

# Run a specific test
pytest tests/test_app.py::test_function_name
```

### Architecture

- **`main.py`** — FastAPI app entry point; registers routers and the `APScheduler` scheduler; exports `job_queue` singleton
- **`config.py`** — All configuration loaded from `.env`; also the single source of truth for `format_model_input()` — both training and inference must use this to keep inputs consistent
- **`classify.py`** — SetFit inference engine; `predict_email()` and `predict_raw_email()`
- **`train.py`** — SetFit training script; reads `.jsonl` files from `TrainingData/` to discover categories dynamically
- **`database.py`** — SQLite persistence via `storage/email_history.db`; handles schema migration inline
- **`imap_client.py`** — Gmail/IMAP integration (fetch, label, scan)
- **`job_queue.py`** — Sequential `JobQueue` class + module-level `job_queue` singleton

#### `jobs/` — Background Job Functions

| Module | Contents |
|--------|----------|
| `jobs/classification.py` | `classification_job` — fetch, classify, label, log |
| `jobs/correction.py` | `check_corrections_job`, `force_check_corrections_job`, `_resolve_correction` |
| `jobs/reclassify.py` | `reclassify_job` — re-run predictions on existing logs |
| `jobs/training_data.py` | `add_to_training_data`, `push_training_data_to_git`, `backfill_training_data_job` |
| `jobs/update.py` | `scheduled_update_job`, `shutdown_server` |

#### `api/` — HTTP API Layer

| Module | Contents |
|--------|----------|
| `api/models.py` | Pydantic request/response models |
| `api/security.py` | `get_api_key` dependency, `api_key_scheme` |
| `api/routes/classification.py` | `POST /run`, `POST /reclassify`, `GET /labels` |
| `api/routes/jobs.py` | `GET /jobs/status`, `POST /jobs/cancel`, `GET /jobs/history` |
| `api/routes/notifications.py` | `GET /notifications`, `POST /notifications/ack`, `POST /notifications/pop`, `GET /notifications/read` |
| `api/routes/admin.py` | `POST /logs/{id}/correction`, `GET /logs/ambiguous`, all `/admin/*` endpoints |
| `api/routes/health.py` | `GET /health`, `GET /stats` |

### Background Jobs (APScheduler → JobQueue)

All recurring jobs are enqueued through `JobQueue` to run sequentially:

| Job | Default | Description |
|-----|---------|-------------|
| Classification | every 5 min | Fetches unprocessed emails from IMAP, classifies, applies labels |
| Re-check | every 12 hr | Detects manual label corrections in Gmail using a gliding-scale age priority |
| Re-classify | every 12 hr (offset) | Re-runs predictions on existing logs with the current model |
| Auto-update | every 1 day | Pushes training data to Git and restarts the service |

### Training Data

Categories are auto-discovered from files in `TrainingData/`. File path relative to `TrainingData/` (minus `.jsonl`) becomes the label — e.g., `TrainingData/WORK/URGENT.jsonl` → label `WORK/URGENT`.

Each `.jsonl` file has one JSON object per line with fields: `subject`, `body`, `from`, `to`, `cc`, `mass_mail` (bool), `attachment_types` (string[]).

### API Security

Protected endpoints require `X-API-Key` header matching `ADMIN_API_KEY` from `.env`. If `ADMIN_API_KEY` is unset, all protected endpoints return 500.

### Testing Pattern

Tests use `pytest` + `FastAPI`'s `TestClient`. The global `JobQueue` worker is stopped in the `stop_queue_worker` autouse fixture, making jobs run synchronously via `job_queue._drain()`. DB is patched to a temp file per test.

Set `TESTING=true`, `ADMIN_API_KEY=testkey`, and `TRAINING_DATA_DIR=<tmpdir>` in env before importing `main`.

---

## UI (Flutter/Riverpod)

### Commands

```bash
# From email_classifier_ui/
flutter pub get
flutter run

# Run on a specific device
flutter run -d macos

# Run tests
flutter test
```

### Configuration

Requires a `.env` file in `email_classifier_ui/` (bundled as a Flutter asset):

```env
API_URL=http://<brain-ip>:8008
API_KEY=<your-admin-api-key>
```

The `.env` file is declared as a Flutter asset in `pubspec.yaml` and loaded at startup via `flutter_dotenv`.

### Architecture

- **`lib/api/api_client.dart`** — `ApiClient` using Dio; reads `API_URL` and `API_KEY` from dotenv at construction
- **`lib/api/models.dart`** — Data models (`Notification`, `StatsResponse`, `RunResponse`, etc.)
- **`lib/providers/api_providers.dart`** — Riverpod providers: `apiClientProvider`, `statsProvider`, `notificationsProvider`, `labelsProvider`
- **`lib/ui/screens/dashboard_screen.dart`** — Single screen, responsive layout (>800px = side-by-side desktop, else stacked mobile)
- **`lib/ui/widgets/`** — `StatsChart` (fl_chart pie/bar), `RecentActivityList` (sliver list with correction support)
- **`lib/ui/theme.dart`** — Dark theme definition

State is managed with Riverpod `FutureProvider.autoDispose`. After mutations (run, reclassify, correct), providers are invalidated with `ref.invalidate()` to trigger a refresh.

---

## Documentation Guidelines

When submitting a PR that changes user-facing behavior, adds new configuration options, or modifies the API, update the relevant README files:

- **`email_classifier_brain/README.md`** — Update when changing environment variables, API endpoints, background jobs, or inference behavior.
- **`email_classifier_ui/README.md`** — Update when changing UI configuration or Flutter setup.
- **Root `README.md`** — Update when adding top-level features visible to all users of the monorepo.
