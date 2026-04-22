"""
main.py
=======
FastAPI application — entry point for the Japan Fire Dashboard.

Starts the web server AND the background ingestion scheduler.
One process does everything.

Run locally:
    uvicorn main:app --reload --port 8000

Then open: http://localhost:8000
"""

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# App lifespan — start/stop scheduler with FastAPI
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start scheduler on startup, stop on shutdown."""
    from dotenv import load_dotenv
    load_dotenv()

    log.info("Japan Fire Dashboard starting ...")

    # Auto-initialise DB on fresh Render deployment
    from db.connection import get_engine
    from db.init_render import init_db_if_needed
    init_db_if_needed(get_engine())

    # Start background ingestion scheduler
    from ingest.scheduler import start_scheduler
    start_scheduler()

    yield  # App is running here

    # Shutdown
    from ingest.scheduler import stop_scheduler
    stop_scheduler()
    log.info("Japan Fire Dashboard stopped.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Japan Forest Fire Dashboard API",
    description="Near real-time wildfire monitoring for Japan — PostGIS + NASA FIRMS + JAXA",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow browser to call API from any origin (needed for MapLibre)
allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)

# Serve static files (index.html map frontend)
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------
@app.head("/", include_in_schema=False)
async def head_root():
    return {}

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    """Serve the map dashboard."""
    html_path = os.path.join("static", "index.html")
    if os.path.exists(html_path):
        with open(html_path, encoding="utf-8") as f:
            return HTMLResponse(f.read())
    return HTMLResponse("""
        <html><body>
        <h2>Japan Fire Dashboard API</h2>
        <p>Frontend not built yet. API endpoints:</p>
        <ul>
          <li><a href="/api/fires">/api/fires</a> — active fire hotspots GeoJSON</li>
          <li><a href="/api/risk">/api/risk</a> — prefecture risk scores GeoJSON</li>
          <li><a href="/api/health">/api/health</a> — system health check</li>
          <li><a href="/docs">/docs</a> — API documentation</li>
        </ul>
        </body></html>
    """)


@app.get("/api/fires")
async def get_fires(
    hours: int = Query(default=48, ge=1, le=240,
                       description="Hours back to fetch (1-240, default 48)"),
    source: str = Query(default=None,
                        description="Filter by source: VIIRS_SNPP, MODIS_Terra, etc."),
    min_confidence: str = Query(default=None,
                                description="Minimum confidence: low, nominal, high"),
    pref_code: str = Query(default=None,
                           description="Filter by prefecture code e.g. '13' for Tokyo"),
):
    """
    Active fire hotspots as GeoJSON FeatureCollection.

    Returns fire detections from NASA FIRMS (MODIS + VIIRS) and JAXA Himawari-9
    for the last N hours over Japan.

    Each feature includes: source, FRP (MW), confidence, prefecture, acquisition time.
    """
    try:
        from db.connection import query_to_df, get_engine
        from sqlalchemy import text

        # Build WHERE clauses
        conditions = ["acq_datetime >= NOW() - (:hours || ' hours')::INTERVAL"]
        params: dict = {"hours": hours}

        if source:
            conditions.append("source = :source")
            params["source"] = source

        if min_confidence:
            conf_order = {"low": 0, "nominal": 1, "high": 2}
            if min_confidence in conf_order:
                allowed = [k for k, v in conf_order.items()
                           if v >= conf_order[min_confidence]]
                conditions.append(f"confidence = ANY(ARRAY{allowed})")

        if pref_code:
            conditions.append("pref_code = :pref_code")
            params["pref_code"] = pref_code


        where = " AND ".join(conditions)

        sql = f"""
            SELECT
                json_build_object(
                    'type', 'FeatureCollection',
                    'generated_at', NOW(),
                    'count', COUNT(*),
                    'features', COALESCE(
                        json_agg(
                            json_build_object(
                                'type', 'Feature',
                                'geometry', ST_AsGeoJSON(geom)::json,
                                'properties', json_build_object(
                                    'id',           id,
                                    'source',       source,
                                    'acq_datetime', acq_datetime,
                                    'frp_mw',       frp_mw,
                                    'confidence',   confidence,
                                    'daynight',     daynight,
                                    'pref_code',    pref_code,
                                    'pref_name',    pref_name_en,
                                    'hours_ago',    ROUND(
                                        EXTRACT(EPOCH FROM (NOW() - acq_datetime))/3600
                                    , 1)
                                )
                            ) ORDER BY acq_datetime DESC
                        ),
                        '[]'::json
                    )
                ) AS geojson
            FROM fire.fire_hotspots
            WHERE {where}
        """

        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(text(sql), params)
            geojson = result.scalar()

        if geojson is None:
            geojson = {"type": "FeatureCollection", "features": [], "count": 0}

        return JSONResponse(content=geojson)

    except Exception as e:
        log.error(f"GET /api/fires error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/risk")
async def get_risk(
    date: str = Query(default=None,
                      description="Date YYYY-MM-DD (default: today)"),
):
    """
    Prefecture fire risk scores as GeoJSON FeatureCollection.

    Returns risk score (0-100), risk level, and hotspot count
    for all 47 Japan prefectures. Used to render the choropleth map layer.
    """
    try:
        from db.connection import query_to_df, get_engine
        from sqlalchemy import text

        if date:
            sql = "SELECT fire.get_risk_geojson(:d::DATE) AS geojson"
            params = {"d": date}
        else:
            sql = "SELECT fire.get_risk_geojson() AS geojson"
            params = {}

        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(text(sql), params)
            geojson = result.scalar()

        return JSONResponse(content=geojson)

    except Exception as e:
        log.error(f"GET /api/risk error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/history")
async def get_history(
    pref_code: str = Query(..., description="Prefecture code e.g. '01' for Hokkaido"),
    days: int = Query(default=30, ge=1, le=365),
):
    """
    Daily fire hotspot count time series for a prefecture.
    Used for the Chart.js time series panel.
    """
    try:
        from db.connection import query_to_df
        sql = """
            SELECT
                acq_date::TEXT AS date,
                COUNT(*) AS hotspot_count,
                COALESCE(SUM(frp_mw), 0) AS total_frp_mw,
                COALESCE(MAX(frp_mw), 0) AS max_frp_mw
            FROM fire.fire_hotspots
            WHERE pref_code = :pref_code
              AND acq_date >= CURRENT_DATE - (:days || ' days')::INTERVAL
            GROUP BY acq_date
            ORDER BY acq_date
        """
        df = query_to_df(sql, {"pref_code": pref_code, "days": days})
        return JSONResponse(content={
            "pref_code": pref_code,
            "days": days,
            "data": df.to_dict(orient="records"),
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/weather")
async def get_weather(
    pref_code: str = Query(default=None,
                           description="Filter by prefecture code"),
):
    """Latest weather observations with Fire Weather Index."""
    try:
        from db.connection import query_to_df

        where = "WHERE obs_datetime >= NOW() - INTERVAL '3 hours'"
        params = {}
        if pref_code:
            where += " AND pref_code = :pref_code"
            params["pref_code"] = pref_code

        sql = f"""
            SELECT DISTINCT ON (station_id) station_id, station_name, pref_code,
                   ST_X(geom) AS longitude, ST_Y(geom) AS latitude,
                   obs_datetime, temp_c, rh_pct,
                   wind_speed_ms, wind_dir_deg, precip_mm,
                   fwi, ffmc, dmc
            FROM fire.weather_observations
            {where}
            ORDER BY station_id, obs_datetime DESC
        """
        df = query_to_df(sql, params)
        records = df.to_dict(orient="records")
        clean = [{k: str(v) if v is not None and not isinstance(v, (int, float, bool)) else v for k, v in r.items()} for r in records]
        return JSONResponse(content={"count": len(df), "observations": clean})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats")
async def get_stats():
    """Summary statistics for dashboard header cards."""
    try:
        from db.connection import query_to_df
        sql = """
            SELECT
                (SELECT COUNT(*) FROM fire.fire_hotspots
                 WHERE acq_datetime >= NOW() - INTERVAL '24 hours') AS fires_24h,
                (SELECT COUNT(*) FROM fire.fire_hotspots
                 WHERE acq_datetime >= NOW() - INTERVAL '1 hour')   AS fires_1h,
                (SELECT COALESCE(SUM(frp_mw),0) FROM fire.fire_hotspots
                 WHERE acq_datetime >= NOW() - INTERVAL '24 hours') AS total_frp_24h,
                (SELECT MAX(acq_datetime) FROM fire.fire_hotspots)  AS latest_detection,
                (SELECT COUNT(DISTINCT pref_code) FROM fire.fire_hotspots
                 WHERE acq_datetime >= NOW() - INTERVAL '24 hours'
                   AND pref_code IS NOT NULL)                       AS prefs_affected,
                (SELECT MAX(started_at) FROM fire.ingestion_log
                 WHERE status = 'success')                          AS last_update
        """
        df = query_to_df(sql)
        row = df.iloc[0].to_dict()
        clean = {k: str(v) if v is not None and not isinstance(v, (int, float, bool)) else v for k, v in row.items()}
        return JSONResponse(content=clean)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check():
    """System health — DB connection, scheduler status, latest data."""
    try:
        from db.connection import check_db_connection
        from ingest.scheduler import get_scheduler_status

        db_stats   = check_db_connection()
        sched_info = get_scheduler_status()

        return JSONResponse(content={
            "status":    "ok",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "database":  db_stats,
            "scheduler": sched_info,
        })
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "detail": str(e)},
        )


@app.post("/api/ingest/trigger", include_in_schema=False)
async def trigger_ingestion():
    """
    Manually trigger an ingestion run.
    Useful for testing without waiting for the scheduler.
    """
    from ingest.scheduler import job_firms_ingestion, job_weather_ingestion
    import asyncio

    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, job_firms_ingestion)
    loop.run_in_executor(None, job_weather_ingestion)

    return JSONResponse(content={
        "status": "triggered",
        "message": "Ingestion jobs started in background",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
