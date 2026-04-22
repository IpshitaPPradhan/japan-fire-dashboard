"""
ingest/firms.py
===============
NASA FIRMS (Fire Information for Resource Management System) ingestion.

Fetches active fire detections for Japan from:
  - MODIS Terra  (MOD14)  — 1km resolution, twice daily
  - MODIS Aqua   (MYD14)  — 1km resolution, twice daily
  - VIIRS SNPP   (VNP14)  — 375m resolution, twice daily  ← primary source
  - VIIRS NOAA20 (VJ114)  — 375m resolution, twice daily

API docs: https://firms.modaps.eosdis.nasa.gov/api/area/
Free key: https://firms.modaps.eosdis.nasa.gov/api/area/

Usage:
    python -m ingest.firms              # fetch last 2 days
    python -m ingest.firms --days 7     # fetch last 7 days
    python -m ingest.firms --source VIIRS_SNPP --days 1
"""

import argparse
import logging
import os
import uuid
from datetime import datetime, timezone
from io import StringIO
from typing import Optional

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Point
from sqlalchemy import text

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Japan bounding box — covers all 4 main islands + Okinawa
# ---------------------------------------------------------------------------
JAPAN_BBOX = "122.9,24.0,153.9,45.5"   # west,south,east,north

# ---------------------------------------------------------------------------
# FIRMS API configuration per satellite source
# ---------------------------------------------------------------------------
FIRMS_SOURCES = {
    "VIIRS_SNPP": {
        "product": "VIIRS_SNPP_NRT",
        "db_source": "VIIRS_SNPP",
        "description": "VIIRS S-NPP 375m (primary — best resolution)",
    },
    "VIIRS_NOAA20": {
        "product": "VIIRS_NOAA20_NRT",
        "db_source": "VIIRS_NOAA20",
        "description": "VIIRS NOAA-20 375m",
    },
    "MODIS_Terra": {
        "product": "MODIS_NRT",
        "db_source": "MODIS_Terra",
        "description": "MODIS Terra 1km",
    },
    "MODIS_Aqua": {
        "product": "MODIS_NRT",
        "db_source": "MODIS_Aqua",
        "description": "MODIS Aqua 1km",
    },
}

# FIRMS base URL
FIRMS_BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"


# ---------------------------------------------------------------------------
# Step 1: Fetch CSV from FIRMS API
# ---------------------------------------------------------------------------

def fetch_firms_csv(
    api_key: str,
    product: str,
    bbox: str,
    days: int = 2,
) -> Optional[pd.DataFrame]:
    """
    Download active fire CSV from NASA FIRMS for a bounding box.

    URL pattern:
        /api/area/csv/{api_key}/{product}/{bbox}/{days}

    Returns a DataFrame or None on failure.
    """
    url = f"{FIRMS_BASE}/{api_key}/{product}/{bbox}/{days}"
    log.info(f"Fetching {product} — last {days} day(s) — {bbox}")
    log.debug(f"URL: {url}")

    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
    except requests.exceptions.HTTPError as e:
        if resp.status_code == 401:
            log.error("FIRMS API key invalid or expired. Get a free key at:")
            log.error("https://firms.modaps.eosdis.nasa.gov/api/area/")
        elif resp.status_code == 429:
            log.error("FIRMS API rate limit hit. Wait 1 hour and retry.")
        else:
            log.error(f"FIRMS HTTP error {resp.status_code}: {e}")
        return None
    except requests.exceptions.ConnectionError:
        log.error("Cannot connect to FIRMS API. Check internet connection.")
        return None
    except requests.exceptions.Timeout:
        log.error("FIRMS API request timed out after 60s.")
        return None

    # Empty response = no fires detected (valid)
    content = resp.text.strip()
    if not content or content == "":
        log.info("No active fires detected in this period.")
        return pd.DataFrame()

    # Parse CSV
    try:
        df = pd.read_csv(StringIO(content))
        log.info(f"Fetched {len(df)} raw fire detections")
        return df
    except Exception as e:
        log.error(f"Failed to parse FIRMS CSV: {e}")
        log.debug(f"Raw response (first 500 chars): {content[:500]}")
        return None


# ---------------------------------------------------------------------------
# Step 2: Clean and standardise the DataFrame
# ---------------------------------------------------------------------------

def clean_firms_df(df: pd.DataFrame, source: str) -> gpd.GeoDataFrame:
    """
    Normalise FIRMS CSV columns to our database schema.

    MODIS columns:  latitude, longitude, brightness, scan, track,
                    acq_date, acq_time, satellite, confidence, version,
                    bright_t31, frp, daynight
    VIIRS columns:  latitude, longitude, bright_ti4, scan, track,
                    acq_date, acq_time, satellite, confidence, version,
                    bright_ti5, frp, daynight
    """
    if df.empty:
        return gpd.GeoDataFrame()

    out = pd.DataFrame()

    # Core spatial
    out["latitude"]  = pd.to_numeric(df["latitude"],  errors="coerce")
    out["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    # Drop rows with invalid coordinates
    out = out.dropna(subset=["latitude", "longitude"])
    out = out[
        (out["latitude"].between(-90, 90)) &
        (out["longitude"].between(-180, 180))
    ]

    # Acquisition datetime — combine acq_date + acq_time (HHMM format)
    def parse_datetime(row):
        try:
            date_str = str(row["acq_date"])           # e.g. "2024-03-15"
            time_str = str(int(row["acq_time"])).zfill(4)  # e.g. "0130"
            dt_str = f"{date_str} {time_str[:2]}:{time_str[2:]}"
            return pd.Timestamp(dt_str, tz="UTC")
        except Exception:
            return pd.NaT

    out["acq_datetime"] = df.apply(parse_datetime, axis=1)
    out["acq_date"]     = out["acq_datetime"].dt.date
    out = out.dropna(subset=["acq_datetime"])

    # Source
    out["source"]    = source
    out["satellite"] = df.get("satellite", pd.Series(dtype=str))

    # Fire radiometric properties
    # MODIS uses 'brightness' + 'bright_t31'; VIIRS uses 'bright_ti4' + 'bright_ti5'
    out["brightness_k"] = pd.to_numeric(
        df.get("brightness", df.get("bright_ti4")), errors="coerce"
    )
    out["bright_t31"] = pd.to_numeric(
        df.get("bright_t31", df.get("bright_ti5")), errors="coerce"
    )
    out["frp_mw"]  = pd.to_numeric(df.get("frp", None),   errors="coerce")
    out["scan"]    = pd.to_numeric(df.get("scan", None),   errors="coerce")
    out["track"]   = pd.to_numeric(df.get("track", None),  errors="coerce")

    # Confidence — normalise to text
    raw_conf = df.get("confidence", pd.Series(dtype=str)).astype(str).str.strip().str.lower()

    def normalise_confidence(val):
        # VIIRS: 'l', 'n', 'h' → expand to full word
        mapping = {"l": "low", "n": "nominal", "h": "high",
                   "low": "low", "nominal": "nominal", "high": "high"}
        return mapping.get(val, val)

    out["confidence"] = raw_conf.apply(normalise_confidence)

    # Numeric confidence (MODIS 0–100)
    out["confidence_pct"] = pd.to_numeric(
        df.get("confidence", None), errors="coerce"
    ).where(lambda x: x.between(0, 100))

    out["version"]  = df.get("version",  pd.Series(dtype=str)).astype(str)
    out["daynight"] = df.get("daynight", pd.Series(dtype=str)).str.upper().str[:1]
    out["daynight"] = out["daynight"].where(out["daynight"].isin(["D", "N"]))

    # Placeholders for prefecture (filled later by spatial join)
    out["pref_code"]    = None
    out["pref_name_en"] = None

    # Build Point geometry
    out["geom"] = out.apply(
        lambda r: Point(r["longitude"], r["latitude"]), axis=1
    )

    gdf = gpd.GeoDataFrame(out, geometry="geom", crs="EPSG:4326")
    log.info(f"Cleaned: {len(gdf)} valid fire detections")
    return gdf


# ---------------------------------------------------------------------------
# Step 3: Prefecture spatial join
# ---------------------------------------------------------------------------

def assign_prefectures(gdf: gpd.GeoDataFrame, engine) -> gpd.GeoDataFrame:
    """
    Spatial join: assign each fire point to a Japan prefecture.
    Points outside Japan (ocean, other countries) get pref_code = None.
    """
    if gdf.empty:
        return gdf

    log.info("Loading prefecture boundaries for spatial join ...")
    pref_gdf = gpd.read_postgis(
        "SELECT pref_code, pref_name_en, geom FROM fire.prefecture_boundaries",
        con=engine.connect(),
        geom_col="geom",
        crs="EPSG:4326",
    )

    # sjoin — fires get the prefecture they fall inside
    joined = gpd.sjoin(
        gdf,
        pref_gdf[["pref_code", "pref_name_en", "geom"]],
        how="left",
        predicate="within",
    )

    # sjoin adds _left/_right suffixes if columns overlap
    if "pref_code_right" in joined.columns:
        joined["pref_code"]    = joined["pref_code_right"]
        joined["pref_name_en"] = joined["pref_name_en_right"]
        joined = joined.drop(columns=[c for c in joined.columns
                                       if c.endswith("_right") or c.endswith("_left")
                                       and c not in ["pref_code", "pref_name_en"]])

    # Drop the sjoin index column
    if "index_right" in joined.columns:
        joined = joined.drop(columns=["index_right"])

    in_japan  = joined["pref_code"].notna().sum()
    total     = len(joined)
    log.info(f"Prefecture join: {in_japan}/{total} points inside Japan")

    return gpd.GeoDataFrame(joined, geometry="geom", crs="EPSG:4326")


# ---------------------------------------------------------------------------
# Step 4: Insert into PostGIS (with deduplication)
# ---------------------------------------------------------------------------

def insert_hotspots(
    gdf: gpd.GeoDataFrame,
    engine,
    batch_id: str,
    source: str,
) -> dict:
    """
    Insert fire hotspots into fire.fire_hotspots.
    Uses ON CONFLICT DO NOTHING to skip duplicates (same source+lat+lon+time).

    Returns dict with inserted/skipped counts.
    """
    if gdf.empty:
        return {"inserted": 0, "skipped": 0}

    inserted = 0
    skipped  = 0

    insert_sql = text("""
        INSERT INTO fire.fire_hotspots (
            geom, latitude, longitude,
            acq_datetime, acq_date,
            source, satellite,
            brightness_k, bright_t31, frp_mw, scan, track,
            confidence, confidence_pct, version, daynight,
            pref_code, pref_name_en,
            ingested_at, batch_id
        ) VALUES (
            ST_SetSRID(ST_MakePoint(:longitude, :latitude), 4326),
            :latitude, :longitude,
            :acq_datetime, :acq_date,
            :source, :satellite,
            :brightness_k, :bright_t31, :frp_mw, :scan, :track,
            :confidence, :confidence_pct, :version, :daynight,
            :pref_code, :pref_name_en,
            NOW(), :batch_id
        )
        ON CONFLICT (source, latitude, longitude, acq_datetime)
        DO NOTHING
    """)

    with engine.connect() as conn:
        for _, row in gdf.iterrows():
            try:
                result = conn.execute(insert_sql, {
                    "latitude":       float(row["latitude"]),
                    "longitude":      float(row["longitude"]),
                    "acq_datetime":   row["acq_datetime"].isoformat(),
                    "acq_date":       str(row["acq_date"]),
                    "source":         str(row["source"]),
                    "satellite":      str(row.get("satellite", "")) or None,
                    "brightness_k":   _safe_float(row.get("brightness_k")),
                    "bright_t31":     _safe_float(row.get("bright_t31")),
                    "frp_mw":         _safe_float(row.get("frp_mw")),
                    "scan":           _safe_float(row.get("scan")),
                    "track":          _safe_float(row.get("track")),
                    "confidence":     str(row.get("confidence", "")) or None,
                    "confidence_pct": _safe_int(row.get("confidence_pct")),
                    "version":        str(row.get("version", "")) or None,
                    "daynight":       str(row.get("daynight", "")) or None,
                    "pref_code":      None if pd.isna(row.get("pref_code")) else str(row.get("pref_code")),
                    "pref_name_en":   None if pd.isna(row.get("pref_name_en")) else str(row.get("pref_name_en")),
                    "batch_id":       batch_id,
                })
                if result.rowcount > 0:
                    inserted += 1
                else:
                    skipped += 1
            except Exception as e:
                log.warning(f"Row insert failed: {e}")
                skipped += 1

        conn.commit()

    log.info(f"Inserted: {inserted} | Skipped (duplicates): {skipped}")
    return {"inserted": inserted, "skipped": skipped}


def _safe_float(val):
    try:
        f = float(val)
        return None if pd.isna(f) else round(f, 4)
    except (TypeError, ValueError):
        return None

def _safe_int(val):
    try:
        f = float(val)
        return None if pd.isna(f) else int(f)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Step 5: Ingestion log
# ---------------------------------------------------------------------------

def log_ingestion(engine, batch_id, source, status,
                  fetched=0, inserted=0, skipped=0,
                  data_start=None, data_end=None,
                  error_msg=None):
    """Write one row to fire.ingestion_log."""
    sql = text("""
        INSERT INTO fire.ingestion_log
            (batch_id, source, pipeline, status,
             records_fetched, records_inserted, records_skipped,
             data_start, data_end, completed_at, error_message)
        VALUES
            (:batch_id, :source, 'ingest.firms', :status,
             :fetched, :inserted, :skipped,
             :data_start, :data_end, NOW(), :error_msg)
    """)
    try:
        with engine.connect() as conn:
            conn.execute(sql, {
                "batch_id":   batch_id,
                "source":     source,
                "status":     status,
                "fetched":    fetched,
                "inserted":   inserted,
                "skipped":    skipped,
                "data_start": data_start,
                "data_end":   data_end,
                "error_msg":  error_msg,
            })
            conn.commit()
    except Exception as e:
        log.warning(f"Could not write ingestion log: {e}")


# ---------------------------------------------------------------------------
# Main pipeline function (called by scheduler + CLI)
# ---------------------------------------------------------------------------

def run_firms_ingestion(
    sources: list = None,
    days: int = 2,
    engine=None,
) -> dict:
    """
    Full FIRMS ingestion pipeline.
    Fetches all configured sources and inserts into PostGIS.

    Args:
        sources: list of source names (default: all 4)
        days:    how many days back to fetch (1–10, free tier max 10)
        engine:  SQLAlchemy engine (created if None)

    Returns:
        summary dict with total counts per source
    """
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv("FIRMS_API_KEY", "").strip()
    if not api_key or api_key == "your_firms_api_key_here":
        log.error("FIRMS_API_KEY not set in .env file!")
        log.error("Get a free key at: https://firms.modaps.eosdis.nasa.gov/api/area/")
        return {"error": "No API key"}

    if engine is None:
        from db.connection import get_engine
        engine = get_engine()

    if sources is None:
        sources = list(FIRMS_SOURCES.keys())

    summary = {}
    batch_id = str(uuid.uuid4())
    log.info(f"Starting FIRMS ingestion — batch {batch_id[:8]}")
    log.info(f"Sources: {sources} | Days back: {days}")

    for source_name in sources:
        cfg = FIRMS_SOURCES[source_name]
        log.info(f"\n--- {source_name} ({cfg['description']}) ---")

        # Fetch
        df = fetch_firms_csv(
            api_key=api_key,
            product=cfg["product"],
            bbox=JAPAN_BBOX,
            days=days,
        )

        if df is None:
            log_ingestion(engine, batch_id, source_name, "failed", error_msg="Fetch failed")
            summary[source_name] = {"status": "failed"}
            continue

        if df.empty:
            log_ingestion(engine, batch_id, source_name, "success",
                          fetched=0, inserted=0, skipped=0)
            summary[source_name] = {"fetched": 0, "inserted": 0}
            continue

        # Clean
        gdf = clean_firms_df(df, cfg["db_source"])
        if gdf.empty:
            log_ingestion(engine, batch_id, source_name, "success")
            summary[source_name] = {"fetched": len(df), "inserted": 0}
            continue

        # Prefecture join
        gdf = assign_prefectures(gdf, engine)

        # Insert
        counts = insert_hotspots(gdf, engine, batch_id, source_name)

        # Log
        data_start = gdf["acq_datetime"].min().isoformat() if not gdf.empty else None
        data_end   = gdf["acq_datetime"].max().isoformat() if not gdf.empty else None
        log_ingestion(
            engine, batch_id, source_name, "success",
            fetched=len(df),
            inserted=counts["inserted"],
            skipped=counts["skipped"],
            data_start=data_start,
            data_end=data_end,
        )
        summary[source_name] = {
            "fetched":  len(df),
            "inserted": counts["inserted"],
            "skipped":  counts["skipped"],
        }

    # Print summary
    log.info("\n" + "="*50)
    log.info("INGESTION SUMMARY")
    log.info("="*50)
    total_inserted = 0
    for src, counts in summary.items():
        if "inserted" in counts:
            log.info(f"  {src:<20} fetched={counts.get('fetched',0):>5}  "
                     f"inserted={counts.get('inserted',0):>5}  "
                     f"skipped={counts.get('skipped',0):>5}")
            total_inserted += counts.get("inserted", 0)
        else:
            log.info(f"  {src:<20} FAILED")
    log.info(f"\n  Total new hotspots inserted: {total_inserted}")

    return summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Fetch NASA FIRMS fire data for Japan")
    parser.add_argument("--days",   type=int, default=2,
                        help="Days back to fetch (1-10, default: 2)")
    parser.add_argument("--source", type=str, default=None,
                        choices=list(FIRMS_SOURCES.keys()),
                        help="Fetch only one source (default: all)")
    args = parser.parse_args()

    sources = [args.source] if args.source else None
    run_firms_ingestion(sources=sources, days=args.days)
