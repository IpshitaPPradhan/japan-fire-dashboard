"""
ingest/scheduler.py
===================
APScheduler configuration — runs all ingestion jobs automatically.
This runs INSIDE the FastAPI process. No separate process needed.

Jobs:
  - FIRMS fire hotspots  → every 1 hour
  - JMA weather          → every 1 hour
  - Daily summary        → every day at 01:00 JST

Usage:
    # Starts automatically when FastAPI starts (imported in main.py)
    # Or test standalone:
    python -m ingest.scheduler
"""

import logging
import os
from datetime import datetime, timezone

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

log = logging.getLogger(__name__)

# Global scheduler instance
scheduler = BackgroundScheduler(timezone="Asia/Tokyo")


# ---------------------------------------------------------------------------
# Job functions — these are what the scheduler actually calls
# ---------------------------------------------------------------------------

def job_firms_ingestion():
    """Fetch latest fire hotspots from NASA FIRMS."""
    log.info(f"[SCHEDULER] FIRMS job starting at {datetime.now(timezone.utc).isoformat()}")
    try:
        from ingest.firms import run_firms_ingestion
        result = run_firms_ingestion(days=2)
        total = sum(v.get("inserted", 0) for v in result.values() if isinstance(v, dict))
        log.info(f"[SCHEDULER] FIRMS job complete — {total} new hotspots inserted")
    except Exception as e:
        log.error(f"[SCHEDULER] FIRMS job FAILED: {e}", exc_info=True)


def job_weather_ingestion():
    """Fetch latest JMA weather observations."""
    log.info(f"[SCHEDULER] Weather job starting at {datetime.now(timezone.utc).isoformat()}")
    try:
        from ingest.weather import run_weather_ingestion
        result = run_weather_ingestion()
        log.info(f"[SCHEDULER] Weather job complete — {result.get('inserted', 0)} stations updated")
    except Exception as e:
        log.error(f"[SCHEDULER] Weather job FAILED: {e}", exc_info=True)

def job_risk_scoring():
    """Compute physics-informed risk scores for all 47 prefectures."""
    log.info("[SCHEDULER] Risk scoring starting ...")
    from ml.risk_model import run_risk_scoring
    result = run_risk_scoring()
    log.info(f"[SCHEDULER] Risk scoring complete — {result.get('scored',0)} prefectures | "
             f"high risk: {result.get('high_risk',0)} | model: {result.get('model','?')}")

def job_daily_summary():
    """
    Daily cleanup and summary job.
    - Logs database stats
    - Could trigger model retraining in Phase 4
    """
    log.info("[SCHEDULER] Daily summary job starting")
    try:
        from db.connection import check_db_connection
        stats = check_db_connection()
        log.info(f"[SCHEDULER] DB stats: {stats.get('hotspots_24h', 0)} hotspots in last 24h")
        log.info(f"[SCHEDULER] Latest hotspot: {stats.get('latest_hotspot')}")
    except Exception as e:
        log.error(f"[SCHEDULER] Daily summary FAILED: {e}", exc_info=True)

def job_risk_scoring():
    """Compute physics-informed risk scores for all 47 prefectures."""
    log.info(f"[SCHEDULER] Risk scoring starting ...")
    from ml.risk_model import run_risk_scoring
    result = run_risk_scoring()
    log.info(f"[SCHEDULER] Risk scoring complete — {result.get('scored', 0)} prefectures | "
             f"high risk: {result.get('high_risk', 0)} | model: {result.get('model', '?')}")


# ---------------------------------------------------------------------------
# Scheduler setup
# ---------------------------------------------------------------------------

def start_scheduler():
    """
    Add all jobs and start the scheduler.
    Called once when FastAPI starts.
    """
    firms_interval = int(os.getenv("FIRMS_POLL_INTERVAL_MIN", "60"))
    weather_interval = int(os.getenv("FIRMS_POLL_INTERVAL_MIN", "60"))

    # FIRMS — every N minutes (default: 60)
    scheduler.add_job(
        job_firms_ingestion,
        trigger=IntervalTrigger(minutes=firms_interval),
        id="firms_ingestion",
        name="NASA FIRMS fire hotspots",
        replace_existing=True,
        max_instances=1,          # never run two at once
        misfire_grace_time=300,   # if missed by <5min, still run
    )
    log.info(f"Scheduled FIRMS ingestion every {firms_interval} minutes")

    # Weather — every hour, offset by 5 minutes
    scheduler.add_job(
        job_weather_ingestion,
        trigger=IntervalTrigger(minutes=weather_interval, start_date=None),
        id="weather_ingestion",
        name="JMA weather observations",
        replace_existing=True,
        max_instances=1,
        misfire_grace_time=300,
    )
    log.info(f"Scheduled weather ingestion every {weather_interval} minutes")

    scheduler.add_job(
        job_risk_scoring,
        trigger=IntervalTrigger(minutes=60),
        id="risk_scoring",
        name="Prefecture risk scoring",
        replace_existing=True,
        next_run_time=datetime.now(timezone.utc) + timedelta(minutes=3),
    )
    log.info("Scheduled risk scoring every 60 minutes")

    

    # Daily summary — 01:00 JST every day
    scheduler.add_job(
        job_daily_summary,
        trigger=CronTrigger(hour=1, minute=0, timezone="Asia/Tokyo"),
        id="daily_summary",
        name="Daily stats summary",
        replace_existing=True,
    )
    log.info("Scheduled daily summary at 01:00 JST")

    # Risk scoring — runs every hour after weather ingestion
    scheduler.add_job(
        job_risk_scoring,
        trigger=IntervalTrigger(minutes=int(os.getenv("FIRMS_POLL_INTERVAL_MIN", 60))),
        id="risk_scoring",
        name="Prefecture risk scoring",
        replace_existing=True,
        next_run_time=datetime.now(timezone.utc) + timedelta(minutes=2),
    )
    log.info("Scheduled risk scoring every 60 minutes")

    scheduler.start()
    log.info("APScheduler started — all ingestion jobs active")

    # Run immediately on startup so dashboard has data right away
    log.info("Running initial ingestion on startup ...")
    job_firms_ingestion()
    job_weather_ingestion()


def stop_scheduler():
    """Gracefully stop scheduler — called when FastAPI shuts down."""
    if scheduler.running:
        scheduler.shutdown(wait=False)
        log.info("APScheduler stopped")


def get_scheduler_status() -> dict:
    """Return status of all scheduled jobs — used by /api/health endpoint."""
    if not scheduler.running:
        return {"running": False, "jobs": []}

    jobs = []
    for job in scheduler.get_jobs():
        jobs.append({
            "id":         job.id,
            "name":       job.name,
            "next_run":   job.next_run_time.isoformat() if job.next_run_time else None,
            "trigger":    str(job.trigger),
        })
    return {"running": True, "jobs": jobs}


# ---------------------------------------------------------------------------
# Standalone test run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    from dotenv import load_dotenv
    load_dotenv()

    log.info("Starting scheduler in test mode (runs once then exits) ...")
    job_firms_ingestion()
    job_weather_ingestion()
    log.info("Test run complete.")
