"""
tests/test_schema.py
====================
Verify the PostGIS schema is correctly set up.
Run with: pytest tests/test_schema.py -v

These tests require a running PostGIS instance with the schema applied.
Set DB_* env vars or DATABASE_URL before running.
"""

import os
import pytest
from sqlalchemy import text

# Skip all tests if no database is configured
pytestmark = pytest.mark.skipif(
    not (os.getenv("DATABASE_URL") or os.getenv("DB_HOST")),
    reason="No database configured — set DATABASE_URL or DB_HOST"
)


@pytest.fixture(scope="module")
def engine():
    from db.connection import get_engine
    return get_engine()


@pytest.fixture(scope="module")
def conn(engine):
    with engine.connect() as connection:
        yield connection


# --- Extension checks ---

def test_postgis_installed(conn):
    result = conn.execute(text("SELECT PostGIS_version()")).scalar()
    assert result is not None
    major = int(result.split(".")[0])
    assert major >= 3, f"Expected PostGIS >= 3, got {result}"


def test_postgis_raster_installed(conn):
    result = conn.execute(text("SELECT PostGIS_Raster_Lib_Version()")).scalar()
    assert result is not None


# --- Table existence ---

EXPECTED_TABLES = [
    "prefecture_boundaries",
    "fire_hotspots",
    "burn_scars",
    "weather_observations",
    "prefecture_risk_scores",
    "ingestion_log",
]

@pytest.mark.parametrize("table_name", EXPECTED_TABLES)
def test_table_exists(conn, table_name):
    result = conn.execute(text(
        "SELECT COUNT(*) FROM information_schema.tables "
        "WHERE table_schema = 'fire' AND table_name = :name"
    ), {"name": table_name}).scalar()
    assert result == 1, f"Table fire.{table_name} does not exist"


# --- View existence ---

EXPECTED_VIEWS = [
    "v_recent_hotspots",
    "v_daily_prefecture_summary",
    "v_fire_clusters",
]

@pytest.mark.parametrize("view_name", EXPECTED_VIEWS)
def test_view_exists(conn, view_name):
    result = conn.execute(text(
        "SELECT COUNT(*) FROM information_schema.views "
        "WHERE table_schema = 'fire' AND table_name = :name"
    ), {"name": view_name}).scalar()
    assert result == 1, f"View fire.{view_name} does not exist"


# --- Spatial indexes ---

def test_hotspot_spatial_index(conn):
    result = conn.execute(text(
        "SELECT COUNT(*) FROM pg_indexes "
        "WHERE schemaname = 'fire' AND tablename = 'fire_hotspots' "
        "AND indexname = 'idx_hotspots_geom'"
    )).scalar()
    assert result == 1, "Spatial index on fire_hotspots.geom not found"


# --- Prefecture data ---

def test_prefectures_loaded(conn):
    count = conn.execute(text(
        "SELECT COUNT(*) FROM fire.prefecture_boundaries"
    )).scalar()
    assert count == 47, f"Expected 47 prefectures, found {count}"


def test_prefecture_geometry_valid(conn):
    invalid = conn.execute(text(
        "SELECT COUNT(*) FROM fire.prefecture_boundaries "
        "WHERE NOT ST_IsValid(geom)"
    )).scalar()
    assert invalid == 0, f"{invalid} prefecture geometries are invalid"


def test_prefecture_geometry_crs(conn):
    """All geometries must be in WGS84 (SRID 4326)."""
    wrong_srid = conn.execute(text(
        "SELECT COUNT(*) FROM fire.prefecture_boundaries "
        "WHERE ST_SRID(geom) != 4326"
    )).scalar()
    assert wrong_srid == 0, f"{wrong_srid} prefectures have wrong SRID"


def test_tokyo_exists(conn):
    result = conn.execute(text(
        "SELECT pref_name_en FROM fire.prefecture_boundaries "
        "WHERE pref_code = '13'"
    )).scalar()
    assert result == "Tokyo"


def test_okinawa_within_japan_bbox(conn):
    """Okinawa should be within the Japan bounding box."""
    result = conn.execute(text("""
        SELECT ST_Within(
            ST_Centroid(geom),
            ST_MakeEnvelope(122.9, 24.0, 153.9, 45.5, 4326)
        )
        FROM fire.prefecture_boundaries WHERE pref_code = '47'
    """)).scalar()
    assert result is True


# --- Functions ---

def test_get_hotspots_geojson_function(conn):
    result = conn.execute(text(
        "SELECT fire.get_hotspots_geojson(48, NULL)"
    )).scalar()
    assert result is not None
    assert result["type"] == "FeatureCollection"
    assert "features" in result


def test_get_risk_geojson_function(conn):
    result = conn.execute(text(
        "SELECT fire.get_risk_geojson()"
    )).scalar()
    assert result is not None
    assert result["type"] == "FeatureCollection"
    # Should have 47 features (one per prefecture)
    assert len(result["features"]) == 47


# --- Constraint checks ---

def test_hotspot_confidence_constraint(conn):
    """INSERT with invalid confidence source should fail."""
    with pytest.raises(Exception):
        conn.execute(text("""
            INSERT INTO fire.fire_hotspots
                (geom, latitude, longitude, acq_datetime, source)
            VALUES
                (ST_SetSRID(ST_MakePoint(139.7, 35.7), 4326),
                 35.7, 139.7, NOW(), 'INVALID_SOURCE')
        """))


def test_prefecture_lookup_function(conn):
    """Point in Tokyo should resolve to pref_code '13'."""
    from db.connection import get_prefecture_for_point
    result = get_prefecture_for_point(35.6895, 139.6917)   # Tokyo
    assert result is not None
    assert result["pref_code"] == "13"
    assert result["pref_name_en"] == "Tokyo"


def test_point_outside_japan_returns_none(conn):
    """Point in the Pacific Ocean should return None."""
    from db.connection import get_prefecture_for_point
    result = get_prefecture_for_point(0.0, 0.0)   # Gulf of Guinea
    assert result is None
