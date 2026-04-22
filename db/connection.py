"""
db/connection.py
================
Database connection and spatial query helpers for the Japan Fire Dashboard.

Provides:
  - Sync engine (SQLAlchemy) for ingestion scripts
  - Async engine (asyncpg via SQLAlchemy) for FastAPI endpoints
  - GeoDataFrame helpers via GeoPandas + GeoAlchemy2
  - Spatial query utilities (bounding box, prefecture lookup, etc.)

Usage:
    from db.connection import get_db, get_engine, spatial_query_to_gdf
"""

import os
import logging
from contextlib import asynccontextmanager, contextmanager
from typing import Optional, Generator
from urllib.parse import quote_plus

import geopandas as gpd
import pandas as pd
from sqlalchemy import create_engine, text, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration — reads from environment variables (set in .env or Render)
# ---------------------------------------------------------------------------

def _build_dsn(async_driver: bool = False) -> str:
    """Build DSN from env vars. Supports DATABASE_URL (Render/Supabase) or parts."""
    
    # Render.com and Supabase both set DATABASE_URL directly
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        # Render provides 'postgres://' — SQLAlchemy needs 'postgresql://'
        database_url = database_url.replace("postgres://", "postgresql://", 1)
        if async_driver:
            # Swap sync driver for async (asyncpg)
            database_url = database_url.replace(
                "postgresql://", "postgresql+asyncpg://", 1
            )
        return database_url

    # Fall back to individual env vars (local dev)
    host     = os.getenv("DB_HOST",     "localhost")
    port     = os.getenv("DB_PORT",     "5432")
    dbname   = os.getenv("DB_NAME",     "japanfire")
    user     = os.getenv("DB_USER",     "postgres")
    password = quote_plus(os.getenv("DB_PASSWORD", "postgres"))

    driver = "postgresql+asyncpg" if async_driver else "postgresql+psycopg2"
    return f"{driver}://{user}:{password}@{host}:{port}/{dbname}"


# ---------------------------------------------------------------------------
# Synchronous engine (ingestion scripts, setup, GeoPandas reads)
# ---------------------------------------------------------------------------

_sync_engine = None

def get_engine():
    """Return (and cache) the synchronous SQLAlchemy engine."""
    global _sync_engine
    if _sync_engine is None:
        dsn = _build_dsn(async_driver=False)
        _sync_engine = create_engine(
            dsn,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,       # reconnect on stale connections
            pool_recycle=1800,        # recycle connections every 30 min
            echo=os.getenv("DB_ECHO", "false").lower() == "true",
        )
        # Ensure PostGIS search_path is set on every new connection
        @event.listens_for(_sync_engine, "connect")
        def set_search_path(dbapi_conn, connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("SET search_path TO fire, public")
            cursor.close()

        log.info("Sync DB engine created")
    return _sync_engine


SyncSession = sessionmaker(autocommit=False, autoflush=False)

@contextmanager
def get_db() -> Generator[Session, None, None]:
    """Context manager for a synchronous DB session."""
    engine = get_engine()
    SyncSession.configure(bind=engine)
    session = SyncSession()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Asynchronous engine (FastAPI endpoints)
# ---------------------------------------------------------------------------

_async_engine = None

def get_async_engine():
    """Return (and cache) the async SQLAlchemy engine for FastAPI."""
    global _async_engine
    if _async_engine is None:
        dsn = _build_dsn(async_driver=True)
        _async_engine = create_async_engine(
            dsn,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            echo=os.getenv("DB_ECHO", "false").lower() == "true",
        )
        log.info("Async DB engine created")
    return _async_engine


AsyncSessionLocal = async_sessionmaker(
    expire_on_commit=False,
    class_=AsyncSession,
)

@asynccontextmanager
async def get_async_db() -> AsyncSession:
    """Async context manager for FastAPI dependency injection."""
    engine = get_async_engine()
    AsyncSessionLocal.configure(bind=engine)
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# FastAPI dependency (use with Depends)
async def db_dependency():
    async with get_async_db() as session:
        yield session


# ---------------------------------------------------------------------------
# Spatial query helpers
# ---------------------------------------------------------------------------

# Japan bounding box (WGS84) — covers all 4 main islands + Okinawa
JAPAN_BBOX = {
    "west":  122.9,
    "south": 24.0,
    "east":  153.9,
    "north": 45.5,
}

JAPAN_BBOX_WKT = (
    f"POLYGON(({JAPAN_BBOX['west']} {JAPAN_BBOX['south']}, "
    f"{JAPAN_BBOX['east']} {JAPAN_BBOX['south']}, "
    f"{JAPAN_BBOX['east']} {JAPAN_BBOX['north']}, "
    f"{JAPAN_BBOX['west']} {JAPAN_BBOX['north']}, "
    f"{JAPAN_BBOX['west']} {JAPAN_BBOX['south']}))"
)


def spatial_query_to_gdf(
    sql: str,
    geom_col: str = "geom",
    crs: str = "EPSG:4326",
    params: Optional[dict] = None,
) -> gpd.GeoDataFrame:
    """
    Run a SQL query and return a GeoDataFrame.
    The query must return a geometry column (default: 'geom').

    Example:
        gdf = spatial_query_to_gdf(
            "SELECT * FROM fire.v_recent_hotspots WHERE pref_code = :pref",
            params={"pref": "13"}
        )
    """
    engine = get_engine()
    return gpd.read_postgis(
        sql=text(sql),
        con=engine.connect(),
        geom_col=geom_col,
        crs=crs,
        params=params or {},
    )


def query_to_df(sql: str, params: Optional[dict] = None) -> pd.DataFrame:
    """Run a SQL query and return a plain DataFrame (no geometry)."""
    engine = get_engine()
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})


def execute(sql: str, params: Optional[dict] = None) -> None:
    """Execute a write statement (INSERT/UPDATE/DELETE)."""
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(text(sql), params or {})
        conn.commit()


# ---------------------------------------------------------------------------
# Common spatial queries used by multiple modules
# ---------------------------------------------------------------------------

def get_prefecture_for_point(lat: float, lon: float) -> Optional[dict]:
    """
    Spatial lookup: which prefecture does this lat/lon fall in?
    Returns dict with pref_code and pref_name_en, or None if outside Japan.
    """
    sql = """
        SELECT pref_code, pref_name_en, pref_name_ja
        FROM fire.prefecture_boundaries
        WHERE ST_Within(
            ST_SetSRID(ST_MakePoint(:lon, :lat), 4326),
            geom
        )
        LIMIT 1
    """
    df = query_to_df(sql, {"lat": lat, "lon": lon})
    if df.empty:
        return None
    return df.iloc[0].to_dict()


def get_hotspots_in_prefecture(pref_code: str, hours: int = 48) -> gpd.GeoDataFrame:
    """Return all fire hotspots in a prefecture within the last N hours."""
    sql = """
        SELECT *
        FROM fire.fire_hotspots
        WHERE pref_code = :pref_code
          AND acq_datetime >= NOW() - (:hours || ' hours')::INTERVAL
        ORDER BY acq_datetime DESC
    """
    return spatial_query_to_gdf(sql, params={"pref_code": pref_code, "hours": hours})


def get_recent_hotspots_geojson(hours: int = 48) -> dict:
    """
    Call the PostGIS function that returns a GeoJSON FeatureCollection.
    This is what the /api/fires endpoint returns directly.
    """
    sql = "SELECT fire.get_hotspots_geojson(:hours, NULL) AS geojson"
    df = query_to_df(sql, {"hours": hours})
    return df.iloc[0]["geojson"]


def get_risk_geojson(date_str: Optional[str] = None) -> dict:
    """
    Return prefecture risk scores as GeoJSON for the choropleth layer.
    date_str: 'YYYY-MM-DD' or None for today.
    """
    if date_str:
        sql = "SELECT fire.get_risk_geojson(:d::DATE) AS geojson"
        params = {"d": date_str}
    else:
        sql = "SELECT fire.get_risk_geojson() AS geojson"
        params = {}
    df = query_to_df(sql, params)
    return df.iloc[0]["geojson"]


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def check_db_connection() -> dict:
    """
    Quick health check — returns DB version + PostGIS version + table row counts.
    Called by GET /api/health endpoint.
    """
    sql = """
        SELECT
            version()                                       AS pg_version,
            PostGIS_version()                               AS postgis_version,
            (SELECT COUNT(*) FROM fire.fire_hotspots
             WHERE acq_datetime >= NOW() - INTERVAL '24 hours') AS hotspots_24h,
            (SELECT COUNT(*) FROM fire.fire_hotspots
             WHERE acq_datetime >= NOW() - INTERVAL '1 hour')   AS hotspots_1h,
            (SELECT MAX(acq_datetime) FROM fire.fire_hotspots)  AS latest_hotspot,
            (SELECT MAX(started_at) FROM fire.ingestion_log
             WHERE status = 'success')                          AS last_successful_ingest
    """
    df = query_to_df(sql)
    return df.iloc[0].to_dict()
