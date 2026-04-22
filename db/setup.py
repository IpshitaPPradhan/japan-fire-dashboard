#!/usr/bin/env python3
"""
db/setup.py
===========
Run this ONCE to initialise the database:
  1. Applies schema.sql  (creates all tables, views, functions)
  2. Loads Japan prefecture boundaries from GSI open GeoJSON
  3. Verifies everything is in order

Usage:
    python -m db.setup
    # or with a custom .env file:
    python -m db.setup --env .env.local
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from sqlalchemy import text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prefecture metadata — JIS X 0401 codes + English names + regions
# ---------------------------------------------------------------------------
PREFECTURE_METADATA = [
    ("01","北海道","Hokkaido","Hokkaido"),
    ("02","青森県","Aomori","Tohoku"),
    ("03","岩手県","Iwate","Tohoku"),
    ("04","宮城県","Miyagi","Tohoku"),
    ("05","秋田県","Akita","Tohoku"),
    ("06","山形県","Yamagata","Tohoku"),
    ("07","福島県","Fukushima","Tohoku"),
    ("08","茨城県","Ibaraki","Kanto"),
    ("09","栃木県","Tochigi","Kanto"),
    ("10","群馬県","Gunma","Kanto"),
    ("11","埼玉県","Saitama","Kanto"),
    ("12","千葉県","Chiba","Kanto"),
    ("13","東京都","Tokyo","Kanto"),
    ("14","神奈川県","Kanagawa","Kanto"),
    ("15","新潟県","Niigata","Chubu"),
    ("16","富山県","Toyama","Chubu"),
    ("17","石川県","Ishikawa","Chubu"),
    ("18","福井県","Fukui","Chubu"),
    ("19","山梨県","Yamanashi","Chubu"),
    ("20","長野県","Nagano","Chubu"),
    ("21","岐阜県","Gifu","Chubu"),
    ("22","静岡県","Shizuoka","Chubu"),
    ("23","愛知県","Aichi","Chubu"),
    ("24","三重県","Mie","Kinki"),
    ("25","滋賀県","Shiga","Kinki"),
    ("26","京都府","Kyoto","Kinki"),
    ("27","大阪府","Osaka","Kinki"),
    ("28","兵庫県","Hyogo","Kinki"),
    ("29","奈良県","Nara","Kinki"),
    ("30","和歌山県","Wakayama","Kinki"),
    ("31","鳥取県","Tottori","Chugoku"),
    ("32","島根県","Shimane","Chugoku"),
    ("33","岡山県","Okayama","Chugoku"),
    ("34","広島県","Hiroshima","Chugoku"),
    ("35","山口県","Yamaguchi","Chugoku"),
    ("36","徳島県","Tokushima","Shikoku"),
    ("37","香川県","Kagawa","Shikoku"),
    ("38","愛媛県","Ehime","Shikoku"),
    ("39","高知県","Kochi","Shikoku"),
    ("40","福岡県","Fukuoka","Kyushu"),
    ("41","佐賀県","Saga","Kyushu"),
    ("42","長崎県","Nagasaki","Kyushu"),
    ("43","熊本県","Kumamoto","Kyushu"),
    ("44","大分県","Oita","Kyushu"),
    ("45","宮崎県","Miyazaki","Kyushu"),
    ("46","鹿児島県","Kagoshima","Kyushu"),
    ("47","沖縄県","Okinawa","Okinawa"),
]

# GSI (国土地理院) open data — Japan prefecture boundaries
# Low-resolution version, good for national-scale choropleth maps
GSI_PREFECTURE_URL = (
    "https://raw.githubusercontent.com/dataofjapan/land/master/japan.geojson"
)

# ---------------------------------------------------------------------------
# Step 1: Apply schema
# ---------------------------------------------------------------------------

def apply_schema(engine) -> None:
    """Read schema.sql and execute it against the database."""
    schema_path = Path(__file__).parent / "schema.sql"
    if not schema_path.exists():
        log.error(f"schema.sql not found at {schema_path}")
        sys.exit(1)

    sql = schema_path.read_text(encoding="utf-8")

    log.info("Applying schema.sql …")
    with engine.connect() as conn:
        # Split on semicolons — psycopg2 can't run multiple statements at once
        statements = [s.strip() for s in sql.split(";") if s.strip()]
        for i, stmt in enumerate(statements):
            try:
                conn.execute(text(stmt))
            except Exception as e:
                log.warning(f"Statement {i+1} warning: {e}")
        conn.commit()
    log.info("Schema applied successfully")


# ---------------------------------------------------------------------------
# Step 2: Load prefecture boundaries
# ---------------------------------------------------------------------------

def load_prefectures(engine) -> None:
    """
    Download Japan prefecture boundaries from GitHub (dataofjapan/land)
    and insert into fire.prefecture_boundaries.

    The GeoJSON uses prefecture names in Japanese — we enrich with our
    metadata table to add English names, JIS codes, and regions.
    """
    log.info("Downloading prefecture boundaries …")

    # Check if already loaded
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT COUNT(*) FROM fire.prefecture_boundaries")
        )
        count = result.scalar()
        if count == 47:
            log.info(f"Prefecture boundaries already loaded ({count} rows) — skipping")
            return
        elif count > 0:
            log.warning(f"Found {count} prefectures (expected 47) — reloading")
            conn.execute(text("TRUNCATE fire.prefecture_boundaries CASCADE"))
            conn.commit()

    # Download GeoJSON
    try:
        resp = requests.get(GSI_PREFECTURE_URL, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        log.error(f"Failed to download prefecture GeoJSON: {e}")
        log.info("Tip: download manually from https://github.com/dataofjapan/land")
        sys.exit(1)

    # Parse to GeoDataFrame
    gdf = gpd.GeoDataFrame.from_features(
        resp.json()["features"],
        crs="EPSG:4326"
    )
    log.info(f"Downloaded {len(gdf)} prefecture features")

    # Build metadata lookup from Japanese name
    # The GeoJSON has 'nam_ja' or 'nam' field — inspect what's available
    log.info(f"GeoJSON columns: {list(gdf.columns)}")

    # Build our metadata lookup by JIS code order (01–47)
    meta_df = pd.DataFrame(
        PREFECTURE_METADATA,
        columns=["pref_code", "pref_name_ja", "pref_name_en", "region"]
    )

    # The dataofjapan GeoJSON uses 'id' field (1-47 as int) for prefecture order
    # We match by index position (GeoJSON is sorted by JIS code)
    if "id" in gdf.columns:
        gdf["pref_code"] = gdf["id"].astype(str).str.zfill(2)
    else:
        # Fallback: assign by row order (GeoJSON is ordered 01–47)
        gdf["pref_code"] = [f"{i+1:02d}" for i in range(len(gdf))]

    # Merge metadata
    gdf = gdf.merge(meta_df, on="pref_code", how="left")

    # Compute area in km² (project to equal-area CRS first)
    gdf_ea = gdf.to_crs("EPSG:6669")   # JGD2011 / Japan Plane Rectangular CS — equal area
    gdf["area_km2"] = (gdf_ea.geometry.area / 1e6).round(2)

    # Ensure MultiPolygon (some prefectures might be Polygon — islands etc.)
    from shapely.geometry import MultiPolygon, Polygon
    def to_multi(geom):
        if isinstance(geom, Polygon):
            return MultiPolygon([geom])
        return geom
    gdf["geometry"] = gdf["geometry"].apply(to_multi)

    # Write to PostGIS
    log.info("Writing prefecture boundaries to PostGIS …")
    write_cols = ["pref_code", "pref_name_ja", "pref_name_en",
                  "region", "area_km2", "geometry"]
    
    gdf_write = gdf[write_cols].copy()
    gdf_write = gdf_write.rename(columns={"geometry": "geom"})
    gdf_write = gdf_write.set_geometry("geom")

    gdf_write.to_postgis(
        name="prefecture_boundaries",
        schema="fire",
        con=engine,
        if_exists="append",
        index=False,
        dtype={"geom": "GEOMETRY"},
    )
    log.info(f"Loaded {len(gdf_write)} prefecture boundaries")


# ---------------------------------------------------------------------------
# Step 3: Create spatial index on prefecture geometry (if not exists)
# ---------------------------------------------------------------------------

def verify_setup(engine) -> None:
    """Run sanity checks and print a summary."""
    log.info("\n" + "="*55)
    log.info("VERIFICATION")
    log.info("="*55)

    checks = [
        ("PostGIS version",
         "SELECT PostGIS_version()"),
        ("Prefecture count",
         "SELECT COUNT(*) FROM fire.prefecture_boundaries"),
        ("Hotspots table exists",
         "SELECT COUNT(*) FROM fire.fire_hotspots"),
        ("Views exist",
         "SELECT COUNT(*) FROM information_schema.views WHERE table_schema='fire'"),
        ("Functions exist",
         "SELECT COUNT(*) FROM information_schema.routines WHERE routine_schema='fire'"),
        ("Tokyo geometry check",
         "SELECT ST_IsValid(geom) FROM fire.prefecture_boundaries WHERE pref_code='13'"),
    ]

    all_ok = True
    with engine.connect() as conn:
        for label, sql in checks:
            try:
                result = conn.execute(text(sql)).scalar()
                log.info(f"  {'OK':<6} {label:<35} → {result}")
            except Exception as e:
                log.error(f"  {'FAIL':<6} {label:<35} → {e}")
                all_ok = False

    # Print prefecture list
    with engine.connect() as conn:
        result = conn.execute(text(
            "SELECT pref_code, pref_name_en, region, ROUND(area_km2) as km2 "
            "FROM fire.prefecture_boundaries ORDER BY pref_code LIMIT 10"
        ))
        rows = result.fetchall()

    log.info("\nFirst 10 prefectures:")
    for row in rows:
        log.info(f"  {row.pref_code}  {row.pref_name_en:<15} {row.region:<12} {row.km2:>8} km²")
    log.info("  … (47 total)")

    if all_ok:
        log.info("\nSetup complete! Database is ready.")
        log.info("Next step: run the FIRMS ingestion to get live fire data:")
        log.info("  python -m ingest.firms")
    else:
        log.error("\nSetup had errors. Check the messages above.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Initialise the Japan Fire Dashboard DB")
    parser.add_argument("--env", default=".env", help="Path to .env file")
    parser.add_argument("--skip-prefectures", action="store_true",
                        help="Skip loading prefecture boundaries")
    parser.add_argument("--schema-only", action="store_true",
                        help="Only apply schema, skip data loading")
    args = parser.parse_args()

    # Load .env file if it exists
    env_path = Path(args.env)
    if env_path.exists():
        from dotenv import load_dotenv
        load_dotenv(env_path)
        log.info(f"Loaded environment from {env_path}")
    else:
        log.info("No .env file found — using existing environment variables")

    # Import here so env vars are set first
    from db.connection import get_engine
    engine = get_engine()

    log.info(f"Connected to database: {os.getenv('DB_NAME', 'japanfire')} "
             f"@ {os.getenv('DB_HOST', 'localhost')}")

    apply_schema(engine)

    if not args.schema_only:
        if not args.skip_prefectures:
            load_prefectures(engine)

    verify_setup(engine)


if __name__ == "__main__":
    main()
