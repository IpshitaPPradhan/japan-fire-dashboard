"""
db/init_render.py
=================
Runs on first Render.com deploy to initialise the database.
Called from main.py lifespan if tables don't exist yet.

This is separate from setup.py (which uses Docker psql).
On Render we apply the schema via Python/SQLAlchemy directly.
"""

import logging
import os
from pathlib import Path
from sqlalchemy import text, inspect

log = logging.getLogger(__name__)


def schema_exists(engine) -> bool:
    """Check if the fire schema and key tables already exist."""
    try:
        with engine.connect() as conn:
            result = conn.execute(text(
                "SELECT COUNT(*) FROM information_schema.tables "
                "WHERE table_schema = 'fire' AND table_name = 'prefecture_boundaries'"
            ))
            return result.scalar() > 0
    except Exception:
        return False


def prefectures_loaded(engine) -> bool:
    """Check if prefecture boundaries are already loaded."""
    try:
        with engine.connect() as conn:
            result = conn.execute(text(
                "SELECT COUNT(*) FROM fire.prefecture_boundaries"
            ))
            return result.scalar() == 47
    except Exception:
        return False


def apply_schema_python(engine) -> None:
    """
    Apply schema using Python/SQLAlchemy (works on Render without psql).
    Reads schema.sql and executes each statement.
    """
    schema_path = Path(__file__).parent / "schema.sql"
    if not schema_path.exists():
        log.error("schema.sql not found")
        return

    sql = schema_path.read_text(encoding="utf-8")

    # Split on semicolons but preserve $$ dollar-quoted blocks
    # Use a simple state machine to avoid splitting inside $$ blocks
    statements = []
    current = []
    in_dollars = False

    for line in sql.split('\n'):
        stripped = line.strip()
        if '$$' in stripped:
            in_dollars = not in_dollars
        current.append(line)
        if not in_dollars and stripped.endswith(';'):
            stmt = '\n'.join(current).strip()
            if stmt and not stmt.startswith('--'):
                statements.append(stmt)
            current = []

    log.info(f"Applying schema ({len(statements)} statements) ...")
    with engine.connect() as conn:
        for stmt in statements:
            try:
                conn.execute(text(stmt.rstrip(';')))
                conn.commit()
            except Exception as e:
                conn.rollback()
                if 'already exists' not in str(e).lower():
                    log.debug(f"Schema stmt warning: {e}")
    log.info("Schema applied")


def load_prefectures_render(engine) -> None:
    """Download and load prefecture boundaries (same logic as setup.py)."""
    import geopandas as gpd
    import pandas as pd
    import requests
    from shapely.geometry import MultiPolygon, Polygon

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

    GSI_URL = "https://raw.githubusercontent.com/dataofjapan/land/master/japan.geojson"

    log.info("Downloading prefecture boundaries ...")
    try:
        resp = requests.get(GSI_URL, timeout=60)
        resp.raise_for_status()
        gdf = gpd.GeoDataFrame.from_features(resp.json()["features"], crs="EPSG:4326")
    except Exception as e:
        log.error(f"Failed to download prefecture GeoJSON: {e}")
        return

    meta_df = pd.DataFrame(PREFECTURE_METADATA,
                           columns=["pref_code","pref_name_ja","pref_name_en","region"])

    if "id" in gdf.columns:
        gdf["pref_code"] = gdf["id"].astype(str).str.zfill(2)
    else:
        gdf["pref_code"] = [f"{i+1:02d}" for i in range(len(gdf))]

    gdf = gdf.merge(meta_df, on="pref_code", how="left")

    gdf_ea = gdf.to_crs("EPSG:6669")
    gdf["area_km2"] = (gdf_ea.geometry.area / 1e6).round(2)

    def to_multi(geom):
        if isinstance(geom, Polygon):
            return MultiPolygon([geom])
        return geom
    gdf["geometry"] = gdf["geometry"].apply(to_multi)

    gdf_write = gdf[["pref_code","pref_name_ja","pref_name_en",
                      "region","area_km2","geometry"]].copy()
    gdf_write = gdf_write.rename(columns={"geometry":"geom"}).set_geometry("geom")

    gdf_write.to_postgis(
        name="prefecture_boundaries", schema="fire",
        con=engine, if_exists="append", index=False,
    )
    log.info(f"Loaded {len(gdf_write)} prefecture boundaries")


def init_db_if_needed(engine) -> None:
    """
    Called on app startup. Initialises DB if this is a fresh Render deployment.
    Does nothing if DB is already set up.
    """
    if schema_exists(engine) and prefectures_loaded(engine):
        log.info("Database already initialised — skipping")
        return

    log.info("Fresh deployment detected — initialising database ...")

    # Enable PostGIS
    with engine.connect() as conn:
        try:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis_raster"))
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS btree_gist"))
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
            conn.commit()
            log.info("PostGIS extensions enabled")
        except Exception as e:
            log.warning(f"Extension setup: {e}")
            conn.rollback()

    if not schema_exists(engine):
        apply_schema_python(engine)

    if not prefectures_loaded(engine):
        load_prefectures_render(engine)

    log.info("Database initialisation complete")
