"""
ingest/weather.py
=================
JMA (Japan Meteorological Agency) weather data ingestion.

Fetches hourly observations from JMA's open data API:
  - Temperature (°C)
  - Relative humidity (%)
  - Wind speed (m/s) and direction (degrees)
  - Precipitation (mm/h)

Then computes a simplified Fire Weather Index (FWI) from the observations.

JMA open data: https://www.jma.go.jp/bosai/en_risk/
AMEDAS stations: https://www.jma.go.jp/bosai/amedas/

Usage:
    python -m ingest.weather
"""

import logging
import os
import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional
import math

import pandas as pd
import requests
from sqlalchemy import text


log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# JMA AMEDAS API endpoints
# ---------------------------------------------------------------------------
# Latest observations for all stations (updated every 10 min)
JMA_AMEDAS_LATEST  = "https://www.jma.go.jp/bosai/amedas/data/latest_time.txt"
JMA_AMEDAS_DATA    = "https://www.jma.go.jp/bosai/amedas/data/point/{station_id}/{datetime_str}_amedas.json"
JMA_AMEDAS_TABLE   = "https://www.jma.go.jp/bosai/amedas/const/amedastable.json"

# ---------------------------------------------------------------------------
# Key JMA AMEDAS stations — one per prefecture, fire-risk focused
# Chosen for coverage of forested/fire-prone areas
# ---------------------------------------------------------------------------
KEY_STATIONS = [
    # (station_id, station_name, pref_code, lat, lon)
    ("14163", "Sapporo",     "01", 43.06, 141.33),
    ("31312", "Aomori",      "02", 40.82, 140.76),
    ("33472", "Morioka",     "03", 39.70, 141.17),
    ("34392", "Sendai",      "04", 38.27, 140.90),
    ("32402", "Akita",       "05", 39.72, 140.10),
    ("35426", "Yamagata",    "06", 38.25, 140.33),
    ("36361", "Fukushima",   "07", 37.75, 140.47),
    ("40201", "Mito",        "08", 36.38, 140.47),
    ("41277", "Utsunomiya",  "09", 36.55, 139.87),
    ("42251", "Maebashi",    "10", 36.40, 139.07),
    ("43056", "Kumagaya",    "11", 36.15, 139.38),
    ("45147", "Choshi",      "12", 35.73, 140.85),
    ("44136", "Tokyo",       "13", 35.69, 139.75),
    ("46106", "Yokohama",    "14", 35.44, 139.65),
    ("54232", "Niigata",     "15", 37.92, 139.05),
    ("55102", "Toyama",      "16", 36.70, 137.22),
    ("56227", "Kanazawa",    "17", 36.58, 136.63),
    ("57066", "Fukui",       "18", 36.07, 136.22),
    ("49142", "Kofu",        "19", 35.67, 138.55),
    ("48156", "Nagano",      "20", 36.66, 138.20),
    ("58362", "Gifu",        "21", 35.42, 136.77),
    ("50331", "Shizuoka",    "22", 34.98, 138.40),
    ("51106", "Nagoya",      "23", 35.17, 136.97),
    ("66356", "Tsu",         "24", 34.73, 136.52),
    ("60821", "Hikone",      "25", 35.27, 136.25),
    ("61286", "Kyoto",       "26", 35.02, 135.73),
    ("62078", "Osaka",       "27", 34.68, 135.52),
    ("63518", "Kobe",        "28", 34.70, 135.18),
    ("65042", "Nara",        "29", 34.68, 135.83),
    ("65356", "Wakayama",    "30", 34.23, 135.17),
    ("69122", "Tottori",     "31", 35.50, 134.23),
    ("68132", "Matsue",      "32", 35.47, 133.05),
    ("66516", "Okayama",     "33", 34.66, 133.92),
    ("67437", "Hiroshima",   "34", 34.40, 132.47),
    ("71106", "Yamaguchi",   "35", 34.17, 131.47),
    ("72086", "Tokushima",   "36", 34.07, 134.57),
    ("72746", "Takamatsu",   "37", 34.32, 134.05),
    ("73166", "Matsuyama",   "38", 33.85, 132.78),
    ("74181", "Kochi",       "39", 33.57, 133.55),
    ("82182", "Fukuoka",     "40", 33.58, 130.38),
    ("84356", "Saga",        "41", 33.27, 130.30),
    ("84766", "Nagasaki",    "42", 32.73, 129.87),
    ("86141", "Kumamoto",    "43", 32.82, 130.70),
    ("83216", "Oita",        "44", 33.23, 131.62),
    ("87376", "Miyazaki",    "45", 31.93, 131.42),
    ("88317", "Kagoshima",   "46", 31.57, 130.55),
    ("91197", "Naha",        "47", 26.20, 127.68),
]


# ---------------------------------------------------------------------------
# Fire Weather Index (simplified Canadian FWI)
# ---------------------------------------------------------------------------

def compute_fwi(temp_c: float, rh_pct: float,
                wind_ms: float, precip_mm: float) -> dict:
    """
    Simplified Fire Weather Index calculation.
    Based on the Canadian FWI System (Van Wagner 1987).

    Returns dict with fwi, ffmc, dmc values.
    All inputs can be None — returns None values gracefully.
    """
    try:
        if any(v is None for v in [temp_c, rh_pct, wind_ms]):
            return {"fwi": None, "ffmc": None, "dmc": None}

        rh  = max(0.0, min(100.0, float(rh_pct)))
        tmp = float(temp_c)
        wnd = max(0.0, float(wind_ms)) * 3.6   # convert m/s → km/h for FWI
        pcp = max(0.0, float(precip_mm or 0))

        # Fine Fuel Moisture Code (FFMC) — simplified
        # Standard initial FFMC = 85
        m0   = 147.2 * (101 - 85) / (59.5 + 85)
        rf   = max(0, pcp - 0.5)
        if rf > 0:
            m0 = m0 - 42.5 * rf * (1 - 100 / (251 - m0)) * (1 - pow(2.718, -6.93 / rf))
        ed   = 0.942 * pow(rh, 0.679) + 11 * pow(2.718, (rh - 100) / 10) + 0.18 * (21.1 - tmp) * (1 - 1/pow(2.718, 0.115 * rh))
        ew   = 0.618 * pow(rh, 0.753) + 10 * pow(2.718, (rh - 100) / 10) + 0.18 * (21.1 - tmp) * (1 - 1/pow(2.718, 0.115 * rh))
        m0   = max(0, min(250, m0))
        if m0 < ed:
            m = ed
        elif m0 > ew:
            m = ew
        else:
            m = m0
        ffmc = 59.5 * (250 - m) / (147.2 + m)
        ffmc = max(0, min(101, ffmc))

        # Duff Moisture Code (DMC) — simplified
        el   = [6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8.0, 7.8, 6.3]
        month = datetime.now(timezone.utc).month
        k    = 1.894 * (tmp + 1.1) * (100 - rh) * el[month - 1] * 1e-6
        dmc  = max(0, 45 + 244.72 * k)

        # Initial Spread Index (ISI)
        fw  = pow(2.718, 0.05039 * wnd)
        fm  = 147.2 * (101 - ffmc) / (59.5 + ffmc)
        sf  = 0.208 * fw * (96 - 8.73 * pow(2.718, -0.1 * fm))
        isi = max(0, sf)

        # Build-up Index (BUI)
        if dmc <= 0.4 * 5:
            bui = 0.8 * dmc * 5 / (dmc + 0.4 * 5)
        else:
            bui = dmc - (1 - 0.8 * 5 / (dmc + 0.4 * 5)) * (0.92 + pow(0.0114 * dmc, 1.7))
        bui = max(0, bui)

        # Fire Weather Index (FWI)
        if bui > 80:
            bb = 0.1 * isi * (1000 / (25 + 108.64 / pow(2.718, 0.023 * bui)))
        else:
            bb = 0.1 * isi * (0.626 * pow(bui, 0.809) + 2)
        fwi = pow(2.718, 2.72 * pow(0.434 * math.log(bb), 0.647)) if bb > 1 else bb
        fwi = max(0, round(fwi, 3))

        return {
            "fwi":  fwi,
            "ffmc": round(ffmc, 3),
            "dmc":  round(dmc, 3),
        }
    except Exception:
        return {"fwi": None, "ffmc": None, "dmc": None}


# ---------------------------------------------------------------------------
# Fetch JMA observations
# ---------------------------------------------------------------------------

def fetch_jma_observations() -> pd.DataFrame:
    """
    Fetch latest JMA AMEDAS observations for our key stations.
    Returns a DataFrame with one row per station.
    """
    log.info("Fetching JMA AMEDAS observations ...")

    # Get the latest data timestamp from JMA
    try:
        resp = requests.get(JMA_AMEDAS_LATEST, timeout=15)
        resp.raise_for_status()
        latest_time = resp.text.strip().replace('"', '')
        log.info(f"JMA latest observation time: {latest_time}")
    except Exception as e:
        log.error(f"Cannot get JMA latest time: {e}")
        # Fall back to current hour
        now = datetime.now(timezone.utc)
        latest_time = now.strftime("%Y%m%d%H%M%S")

    # Fetch the full AMEDAS dataset for that timestamp
    try:
        # JMA provides data as: /data/map/{YYYYMMDDHHMMSS}_amedas.json
        
        # Convert JMA's ISO timestamp to the YYYYMMDDHHMMSS format it expects
        
        from datetime import datetime
        # Keep JST as-is (JMA uses JST in URLs), format as YYYYMMDDHHmm + "00"
        clean = datetime.fromisoformat(latest_time).strftime("%Y%m%d%H%M") + "00"
        data_url = f"https://www.jma.go.jp/bosai/amedas/data/map/{clean}.json"
        resp = requests.get(data_url, timeout=30)
        resp.raise_for_status()
        raw_data = resp.json()
    except Exception as e:
        log.error(f"Cannot fetch JMA AMEDAS data: {e}")
        return pd.DataFrame()

    # Parse our key stations
    rows = []
    obs_dt = datetime.now(timezone.utc)  # approximate

    for station_id, station_name, pref_code, lat, lon in KEY_STATIONS:
        station_data = raw_data.get(station_id, {})
        if not station_data:
            continue

        def get_val(key):
            """Extract numeric value from JMA format: [value, quality_flag]"""
            v = station_data.get(key)
            if v and isinstance(v, list) and len(v) > 0:
                try:
                    return float(v[0])
                except (TypeError, ValueError):
                    return None
            return None

        temp_c      = get_val("temp")
        rh_pct      = get_val("humidity")
        wind_speed  = get_val("wind")        # m/s
        wind_dir    = get_val("windDirection")  # 16-point compass → degrees
        precip_mm   = get_val("precipitation10m")  # 10-min precip

        # Convert wind direction from 16-point (JMA) to degrees
        if wind_dir is not None:
            wind_dir_deg = wind_dir * 22.5  # 1=NNE=22.5°, etc.
        else:
            wind_dir_deg = None

        # Scale 10-min precip to hourly
        if precip_mm is not None:
            precip_mm_hr = precip_mm * 6
        else:
            precip_mm_hr = None

        # Compute Fire Weather Index
        fwi_vals = compute_fwi(temp_c, rh_pct, wind_speed, precip_mm_hr)

        rows.append({
            "station_id":    station_id,
            "station_name":  station_name,
            "pref_code":     pref_code,
            "latitude":      lat,
            "longitude":     lon,
            "obs_datetime":  obs_dt,
            "temp_c":        temp_c,
            "rh_pct":        rh_pct,
            "wind_speed_ms": wind_speed,
            "wind_dir_deg":  wind_dir_deg,
            "precip_mm":     precip_mm_hr,
            "dewpoint_c":    None,   # not in AMEDAS basic
            "fwi":           fwi_vals["fwi"],
            "ffmc":          fwi_vals["ffmc"],
            "dmc":           fwi_vals["dmc"],
        })

    log.info(f"Parsed {len(rows)} station observations")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Insert weather observations
# ---------------------------------------------------------------------------

def insert_weather(df: pd.DataFrame, engine) -> dict:
    """Insert weather observations into fire.weather_observations."""
    if df.empty:
        return {"inserted": 0, "skipped": 0}

    sql = text("""
        INSERT INTO fire.weather_observations (
            geom, station_id, station_name, pref_code,
            obs_datetime, temp_c, rh_pct,
            wind_speed_ms, wind_dir_deg, precip_mm,
            dewpoint_c, fwi, ffmc, dmc, ingested_at
        ) VALUES (
            ST_SetSRID(ST_MakePoint(:lon, :lat), 4326),
            :station_id, :station_name, :pref_code,
            :obs_datetime, :temp_c, :rh_pct,
            :wind_speed_ms, :wind_dir_deg, :precip_mm,
            :dewpoint_c, :fwi, :ffmc, :dmc, NOW()
        )
        ON CONFLICT (station_id, obs_datetime) DO UPDATE SET
            temp_c        = EXCLUDED.temp_c,
            rh_pct        = EXCLUDED.rh_pct,
            wind_speed_ms = EXCLUDED.wind_speed_ms,
            wind_dir_deg  = EXCLUDED.wind_dir_deg,
            precip_mm     = EXCLUDED.precip_mm,
            fwi           = EXCLUDED.fwi,
            ffmc          = EXCLUDED.ffmc,
            dmc           = EXCLUDED.dmc,
            ingested_at   = NOW()
    """)

    inserted = skipped = 0
    with engine.connect() as conn:
        for _, row in df.iterrows():
            try:
                conn.execute(sql, {
                    "lat":          row["latitude"],
                    "lon":          row["longitude"],
                    "station_id":   row["station_id"],
                    "station_name": row["station_name"],
                    "pref_code":    row["pref_code"],
                    "obs_datetime": row["obs_datetime"].isoformat(),
                    "temp_c":       row.get("temp_c"),
                    "rh_pct":       row.get("rh_pct"),
                    "wind_speed_ms":row.get("wind_speed_ms"),
                    "wind_dir_deg": row.get("wind_dir_deg"),
                    "precip_mm":    row.get("precip_mm"),
                    "dewpoint_c":   row.get("dewpoint_c"),
                    "fwi":          row.get("fwi"),
                    "ffmc":         row.get("ffmc"),
                    "dmc":          row.get("dmc"),
                })
                inserted += 1
            except Exception as e:
                log.warning(f"Weather insert failed for {row['station_name']}: {e}")
                skipped += 1
        conn.commit()

    log.info(f"Weather: inserted/updated {inserted} | failed {skipped}")
    return {"inserted": inserted, "skipped": skipped}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_weather_ingestion(engine=None) -> dict:
    """Full weather ingestion pipeline."""
    if engine is None:
        from db.connection import get_engine
        engine = get_engine()

    df  = fetch_jma_observations()
    if df.empty:
        log.warning("No weather data fetched")
        return {"inserted": 0}

    counts = insert_weather(df, engine)
    return counts


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    from dotenv import load_dotenv
    load_dotenv()
    run_weather_ingestion()
