"""
tests/test_firms.py
===================
Tests for the NASA FIRMS ingestion pipeline.

Run with: pytest tests/test_firms.py -v

Tests that don't need a FIRMS API key run always.
Tests that need the API key are skipped if it's not set.
"""

import os
import pytest
import pandas as pd
from io import StringIO
from unittest.mock import patch, MagicMock

# ---------------------------------------------------------------------------
# Sample FIRMS CSV data (real format from VIIRS SNPP)
# ---------------------------------------------------------------------------
SAMPLE_VIIRS_CSV = """latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_ti5,frp,daynight
35.678,139.712,334.2,0.38,0.36,2024-03-15,0130,N20,VIIRS,n,2.0NRT,290.1,12.5,D
34.123,135.456,342.1,0.38,0.36,2024-03-15,0130,N20,VIIRS,h,2.0NRT,295.2,28.3,D
43.567,141.890,328.5,0.38,0.36,2024-03-15,0245,N20,VIIRS,l,2.0NRT,285.6,8.1,N
33.891,130.234,355.7,0.38,0.36,2024-03-15,0130,N20,VIIRS,n,2.0NRT,300.4,45.2,D
"""

SAMPLE_MODIS_CSV = """latitude,longitude,brightness,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_t31,frp,daynight
35.678,139.712,315.4,1.0,1.0,2024-03-15,0130,Terra,MODIS,82,6.1NRT,290.1,15.2,D
34.123,135.456,328.9,1.0,1.0,2024-03-15,0130,Terra,MODIS,95,6.1NRT,295.2,32.1,D
"""

OUTSIDE_JAPAN_CSV = """latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_ti5,frp,daynight
0.0,0.0,334.2,0.38,0.36,2024-03-15,0130,N20,VIIRS,n,2.0NRT,290.1,12.5,D
51.5,-0.1,342.1,0.38,0.36,2024-03-15,0130,N20,VIIRS,h,2.0NRT,295.2,28.3,D
"""


# ---------------------------------------------------------------------------
# Test: CSV parsing and cleaning
# ---------------------------------------------------------------------------

class TestCleanFirmsDF:
    """Test the clean_firms_df function with sample data."""

    def test_viirs_parsing(self):
        from ingest.firms import clean_firms_df
        df = pd.read_csv(StringIO(SAMPLE_VIIRS_CSV))
        gdf = clean_firms_df(df, "VIIRS_SNPP")

        assert len(gdf) == 4
        assert "latitude" in gdf.columns
        assert "longitude" in gdf.columns
        assert "frp_mw" in gdf.columns
        assert "acq_datetime" in gdf.columns
        assert "geom" in gdf.columns

    def test_modis_parsing(self):
        from ingest.firms import clean_firms_df
        df = pd.read_csv(StringIO(SAMPLE_MODIS_CSV))
        gdf = clean_firms_df(df, "MODIS_Terra")

        assert len(gdf) == 2
        assert all(gdf["source"] == "MODIS_Terra")

    def test_confidence_normalised(self):
        """VIIRS 'n' should become 'nominal', 'h' → 'high', 'l' → 'low'."""
        from ingest.firms import clean_firms_df
        df = pd.read_csv(StringIO(SAMPLE_VIIRS_CSV))
        gdf = clean_firms_df(df, "VIIRS_SNPP")

        assert "nominal" in gdf["confidence"].values
        assert "high"    in gdf["confidence"].values
        assert "low"     in gdf["confidence"].values
        assert "n"       not in gdf["confidence"].values

    def test_geometry_created(self):
        """All rows should have valid Point geometry."""
        from ingest.firms import clean_firms_df
        df = pd.read_csv(StringIO(SAMPLE_VIIRS_CSV))
        gdf = clean_firms_df(df, "VIIRS_SNPP")

        assert gdf.crs.to_epsg() == 4326
        assert all(gdf.geometry.geom_type == "Point")
        assert all(gdf.geometry.is_valid)

    def test_datetime_parsed(self):
        """acq_datetime should be timezone-aware UTC timestamps."""
        from ingest.firms import clean_firms_df
        df = pd.read_csv(StringIO(SAMPLE_VIIRS_CSV))
        gdf = clean_firms_df(df, "VIIRS_SNPP")

        assert gdf["acq_datetime"].dtype == "datetime64[ns, UTC]"
        assert all(gdf["acq_datetime"].notna())

    def test_empty_df(self):
        """Empty input should return empty GeoDataFrame gracefully."""
        from ingest.firms import clean_firms_df
        gdf = clean_firms_df(pd.DataFrame(), "VIIRS_SNPP")
        assert gdf.empty

    def test_frp_numeric(self):
        """FRP values should be numeric, not strings."""
        from ingest.firms import clean_firms_df
        df = pd.read_csv(StringIO(SAMPLE_VIIRS_CSV))
        gdf = clean_firms_df(df, "VIIRS_SNPP")

        assert pd.api.types.is_float_dtype(gdf["frp_mw"])


# ---------------------------------------------------------------------------
# Test: Fire Weather Index calculation
# ---------------------------------------------------------------------------

class TestFWI:

    def test_fwi_high_risk(self):
        """High temp, low humidity, high wind → high FWI."""
        from ingest.weather import compute_fwi
        result = compute_fwi(temp_c=35, rh_pct=15, wind_ms=10, precip_mm=0)
        assert result["fwi"] is not None
        assert result["fwi"] > 0
        assert result["ffmc"] is not None

    def test_fwi_low_risk(self):
        """Low temp, high humidity, no wind → low FWI."""
        from ingest.weather import compute_fwi
        result = compute_fwi(temp_c=10, rh_pct=90, wind_ms=0, precip_mm=5)
        assert result["fwi"] is not None

    def test_fwi_none_inputs(self):
        """None inputs should return None values gracefully."""
        from ingest.weather import compute_fwi
        result = compute_fwi(temp_c=None, rh_pct=None, wind_ms=None, precip_mm=None)
        assert result["fwi"] is None
        assert result["ffmc"] is None

    def test_fwi_increases_with_temperature(self):
        """Higher temperature should give higher FWI (all else equal)."""
        from ingest.weather import compute_fwi
        low  = compute_fwi(temp_c=10, rh_pct=50, wind_ms=5, precip_mm=0)
        high = compute_fwi(temp_c=35, rh_pct=50, wind_ms=5, precip_mm=0)
        assert high["fwi"] > low["fwi"]


# ---------------------------------------------------------------------------
# Test: Japan bounding box
# ---------------------------------------------------------------------------

class TestJapanBbox:

    def test_bbox_format(self):
        """FIRMS bbox string should be valid west,south,east,north."""
        from ingest.firms import JAPAN_BBOX
        parts = [float(x) for x in JAPAN_BBOX.split(",")]
        west, south, east, north = parts
        assert west  < east    # west < east
        assert south < north   # south < north
        assert 120 < west  < 130    # western Japan boundary
        assert 20  < south < 30     # Okinawa southernmost
        assert 140 < east  < 160    # eastern Japan boundary
        assert 40  < north < 50     # Hokkaido northernmost

    def test_tokyo_in_bbox(self):
        """Tokyo coordinates should be within the Japan bbox."""
        from ingest.firms import JAPAN_BBOX
        west, south, east, north = [float(x) for x in JAPAN_BBOX.split(",")]
        tokyo_lat, tokyo_lon = 35.6895, 139.6917
        assert west  <= tokyo_lon <= east
        assert south <= tokyo_lat <= north


# ---------------------------------------------------------------------------
# Integration test (requires DB + API key)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.getenv("FIRMS_API_KEY") or
    os.getenv("FIRMS_API_KEY") == "your_firms_api_key_here",
    reason="FIRMS_API_KEY not set"
)
@pytest.mark.skipif(
    not (os.getenv("DATABASE_URL") or os.getenv("DB_HOST")),
    reason="No database configured"
)
def test_live_firms_fetch():
    """
    Live integration test — fetches 1 day of real VIIRS data for Japan.
    Only runs if FIRMS_API_KEY and DB are configured.
    """
    from ingest.firms import fetch_firms_csv, JAPAN_BBOX
    api_key = os.getenv("FIRMS_API_KEY")

    df = fetch_firms_csv(api_key, "VIIRS_SNPP_NRT", JAPAN_BBOX, days=1)
    assert df is not None   # None means API error
    # Empty DataFrame is valid — no fires today is possible
    if not df.empty:
        assert "latitude"  in df.columns
        assert "longitude" in df.columns
        assert "frp"       in df.columns
