import datetime
import logging

from contextlib import asynccontextmanager
from fastapi import FastAPI
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

import config
import database
from job_queue import job_queue  # re-exported for backward compat with tests
from jobs.classification import classification_job
from jobs.correction import check_corrections_job
from jobs.reclassify import reclassify_job
from jobs.update import scheduled_update_job
from api.routes import classification, jobs, notifications, admin, health

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

scheduler = BackgroundScheduler()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Initializing database...")
    database.init_db()

    # Wrapper functions for the scheduler to enqueue jobs
    def _enqueue_classification():
        job_queue.enqueue("classification", classification_job)

    def _enqueue_recheck():
        job_queue.enqueue("recheck", check_corrections_job)

    def _enqueue_reclassify():
        job_queue.enqueue("reclassify", reclassify_job)

    # Start scheduler
    logger.info("Starting scheduler...")
    if config.ENABLE_AUTO_CLASSIFICATION:
        scheduler.add_job(
            _enqueue_classification,
            trigger=IntervalTrigger(minutes=5),
            id="classification_job",
            replace_existing=True
        )
    else:
        logger.info("Automatic classification is disabled via ENABLE_AUTO_CLASSIFICATION.")

    if config.ENABLE_RECHECK_JOB:
        next_run = datetime.datetime.now() + datetime.timedelta(minutes=2)
        scheduler.add_job(
            _enqueue_recheck,
            trigger=IntervalTrigger(hours=config.RECHECK_INTERVAL_HOURS),
            id="check_corrections_job",
            replace_existing=True,
            next_run_time=next_run
        )
    else:
        logger.info("Re-check job disabled.")

    if config.ENABLE_RECLASSIFY_JOB:
        reclassify_offset = datetime.timedelta(hours=config.RECLASSIFY_INTERVAL_HOURS / 2)
        scheduler.add_job(
            _enqueue_reclassify,
            trigger=IntervalTrigger(hours=config.RECLASSIFY_INTERVAL_HOURS),
            id="reclassify_job",
            replace_existing=True,
            next_run_time=datetime.datetime.now() + reclassify_offset,
        )
    else:
        logger.info("Reclassify job disabled.")

    scheduler.add_job(
        scheduled_update_job,
        trigger=IntervalTrigger(days=1),
        id="auto_update_job",
        replace_existing=True
    )
    scheduler.start()
    logger.info("Scheduler started.")

    yield

    # Shutdown
    scheduler.shutdown()
    job_queue.shutdown()
    logger.info("Scheduler and JobQueue shutdown.")


app = FastAPI(title="Email Classifier Microservice", lifespan=lifespan)

# Register routers
app.include_router(classification.router)
app.include_router(jobs.router)
app.include_router(notifications.router)
app.include_router(admin.router)
app.include_router(health.router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8008)
