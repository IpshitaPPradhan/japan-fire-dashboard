-- =============================================================
-- Japan Forest Fire Dashboard — PostGIS Schema
-- =============================================================
-- Run once to initialise the database.
-- Requires PostgreSQL >= 14 with PostGIS >= 3.x
-- =============================================================

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_raster;
CREATE EXTENSION IF NOT EXISTS btree_gist;    -- for temporal exclusion constraints
CREATE EXTENSION IF NOT EXISTS pg_trgm;       -- for fast text search on prefecture names

-- =============================================================
-- SCHEMA
-- =============================================================
CREATE SCHEMA IF NOT EXISTS fire;
SET search_path TO fire, public;

-- =============================================================
-- 1. REFERENCE: Japan prefecture boundaries
--    Populated once from GSI GeoJSON (see ingest/load_prefectures.py)
-- =============================================================
CREATE TABLE IF NOT EXISTS fire.prefecture_boundaries (
    id              SERIAL PRIMARY KEY,
    pref_code       CHAR(2)      NOT NULL UNIQUE,   -- JIS X 0401 (01–47)
    pref_name_ja    TEXT         NOT NULL,           -- 北海道, 東京都, etc.
    pref_name_en    TEXT         NOT NULL,           -- Hokkaido, Tokyo, etc.
    region          TEXT         NOT NULL,           -- Hokkaido / Tohoku / Kanto / ...
    area_km2        NUMERIC(10,2),
    geom            GEOMETRY(MULTIPOLYGON, 4326) NOT NULL,
    created_at      TIMESTAMPTZ  DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_pref_geom
    ON fire.prefecture_boundaries USING GIST (geom);
CREATE INDEX IF NOT EXISTS idx_pref_code
    ON fire.prefecture_boundaries (pref_code);
CREATE INDEX IF NOT EXISTS idx_pref_name_trgm
    ON fire.prefecture_boundaries USING GIN (pref_name_en gin_trgm_ops);

-- =============================================================
-- 2. ACTIVE FIRE HOTSPOTS
--    Ingested hourly from NASA FIRMS (MODIS Terra/Aqua + VIIRS S-NPP/NOAA-20)
--    and JAXA Himawari-9 wildfire product
-- =============================================================
CREATE TABLE IF NOT EXISTS fire.fire_hotspots (
    id              BIGSERIAL    PRIMARY KEY,

    -- Spatial
    geom            GEOMETRY(POINT, 4326) NOT NULL,
    latitude        NUMERIC(8,5) NOT NULL,
    longitude       NUMERIC(8,5) NOT NULL,

    -- Temporal
    acq_datetime    TIMESTAMPTZ  NOT NULL,   -- acquisition UTC datetime
    acq_date        DATE         NOT NULL, 
    -- Source metadata
    source          TEXT         NOT NULL CHECK (source IN (
                        'MODIS_Terra', 'MODIS_Aqua',
                        'VIIRS_SNPP', 'VIIRS_NOAA20',
                        'HIMAWARI9'
                    )),
    satellite       TEXT,                    -- raw satellite string from API

    -- Fire radiometric properties
    brightness_k    NUMERIC(7,2),            -- MODIS: brightness temp band 21/22 (K)
    bright_t31      NUMERIC(7,2),            -- MODIS: brightness temp band 31 (K)
    frp_mw          NUMERIC(10,3),           -- Fire Radiative Power (MW) — key intensity metric
    scan            NUMERIC(5,3),            -- MODIS scan pixel size (km)
    track           NUMERIC(5,3),            -- MODIS track pixel size (km)

    -- Quality / confidence
    confidence      TEXT,                    -- MODIS: 'low'|'nominal'|'high'; VIIRS: 'l'|'n'|'h'
    confidence_pct  SMALLINT,                -- MODIS: 0-100 numeric confidence
    version         TEXT,                    -- FIRMS version string
    daynight        CHAR(1) CHECK (daynight IN ('D','N')),

    -- Prefecture join (denormalised for fast queries)
    pref_code       CHAR(2) REFERENCES fire.prefecture_boundaries(pref_code),
    pref_name_en    TEXT,

    -- Ingestion bookkeeping
    ingested_at     TIMESTAMPTZ  DEFAULT NOW(),
    batch_id        UUID,         -- groups one ingestion run together

    -- Dedup: same source + location + time = same fire pixel
    CONSTRAINT uq_hotspot UNIQUE (source, latitude, longitude, acq_datetime)
);

-- Spatial index — most queries filter by bounding box
CREATE INDEX IF NOT EXISTS idx_hotspots_geom
    ON fire.fire_hotspots USING GIST (geom);

-- Time-based queries (dashboards, time sliders)
CREATE INDEX IF NOT EXISTS idx_hotspots_acq_datetime
    ON fire.fire_hotspots (acq_datetime DESC);

-- Per-prefecture time series
CREATE INDEX IF NOT EXISTS idx_hotspots_pref_date
    ON fire.fire_hotspots (pref_code, acq_date DESC);

-- Source filter (compare MODIS vs VIIRS coverage)
CREATE INDEX IF NOT EXISTS idx_hotspots_source
    ON fire.fire_hotspots (source, acq_datetime DESC);

-- =============================================================
-- 3. BURN SCARS (POLYGON)
--    Derived from Sentinel-2 dNBR — updated weekly
-- =============================================================
CREATE TABLE IF NOT EXISTS fire.burn_scars (
    id              BIGSERIAL    PRIMARY KEY,
    geom            GEOMETRY(MULTIPOLYGON, 4326) NOT NULL,

    -- Detection window
    pre_date        DATE         NOT NULL,   -- Sentinel-2 pre-fire scene date
    post_date       DATE         NOT NULL,   -- Sentinel-2 post-fire scene date

    -- Severity classification from dNBR thresholds
    -- (Key et al. 2006: unburned <0.1, low 0.1–0.27, moderate 0.27–0.66, high >0.66)
    severity_class  TEXT         NOT NULL CHECK (severity_class IN (
                        'unburned', 'low', 'moderate', 'high', 'very_high'
                    )),
    mean_dnbr       NUMERIC(6,4),           -- mean differenced NBR over polygon
    max_dnbr        NUMERIC(6,4),
    area_ha         NUMERIC(12,2),          -- burned area in hectares

    -- Satellite metadata
    sentinel2_tile  TEXT,                   -- e.g. '54SUE' (MGRS tile)
    cloud_cover_pct NUMERIC(5,2),

    -- Prefecture linkage
    pref_code       CHAR(2) REFERENCES fire.prefecture_boundaries(pref_code),
    pref_name_en    TEXT,

    -- Source fire event linkage (optional)
    linked_hotspot_ids  BIGINT[],

    ingested_at     TIMESTAMPTZ  DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_burn_scars_geom
    ON fire.burn_scars USING GIST (geom);
CREATE INDEX IF NOT EXISTS idx_burn_scars_post_date
    ON fire.burn_scars (post_date DESC);
CREATE INDEX IF NOT EXISTS idx_burn_scars_severity
    ON fire.burn_scars (severity_class, post_date DESC);

-- =============================================================
-- 4. WEATHER OBSERVATIONS
--    Ingested from JMA open data API — hourly
-- =============================================================
CREATE TABLE IF NOT EXISTS fire.weather_observations (
    id              BIGSERIAL    PRIMARY KEY,
    geom            GEOMETRY(POINT, 4326) NOT NULL,

    station_id      TEXT         NOT NULL,   -- JMA station code
    station_name    TEXT         NOT NULL,
    pref_code       CHAR(2)      REFERENCES fire.prefecture_boundaries(pref_code),

    obs_datetime    TIMESTAMPTZ  NOT NULL,

    -- Fire-relevant meteorological variables
    temp_c          NUMERIC(5,2),            -- air temperature (°C)
    rh_pct          NUMERIC(5,2),            -- relative humidity (%)
    wind_speed_ms   NUMERIC(6,2),            -- wind speed (m/s)
    wind_dir_deg    NUMERIC(5,1),            -- wind direction (degrees, 0=N)
    precip_mm       NUMERIC(7,2),            -- precipitation (mm/h)
    dewpoint_c      NUMERIC(5,2),

    -- Derived fire weather
    fwi             NUMERIC(7,3),            -- Fire Weather Index (Canadian FWI)
    ffmc            NUMERIC(7,3),            -- Fine Fuel Moisture Code
    dmc             NUMERIC(7,3),            -- Duff Moisture Code

    ingested_at     TIMESTAMPTZ  DEFAULT NOW(),

    CONSTRAINT uq_weather UNIQUE (station_id, obs_datetime)
);

CREATE INDEX IF NOT EXISTS idx_weather_geom
    ON fire.weather_observations USING GIST (geom);
CREATE INDEX IF NOT EXISTS idx_weather_datetime
    ON fire.weather_observations (obs_datetime DESC);
CREATE INDEX IF NOT EXISTS idx_weather_pref
    ON fire.weather_observations (pref_code, obs_datetime DESC);

-- =============================================================
-- 5. PREFECTURE RISK SCORES
--    Updated daily by ML model (XGBoost)
--    One row per prefecture per day
-- =============================================================
CREATE TABLE IF NOT EXISTS fire.prefecture_risk_scores (
    id              BIGSERIAL    PRIMARY KEY,
    pref_code       CHAR(2)      NOT NULL REFERENCES fire.prefecture_boundaries(pref_code),
    pref_name_en    TEXT         NOT NULL,
    score_date      DATE         NOT NULL,

    -- Overall risk (0–100)
    risk_score      NUMERIC(5,2) NOT NULL CHECK (risk_score BETWEEN 0 AND 100),
    risk_level      TEXT         NOT NULL CHECK (risk_level IN (
                        'very_low','low','moderate','high','very_high','extreme'
                    )),

    -- Feature contributions (SHAP values stored for explainability)
    feature_ndvi_anomaly    NUMERIC(6,4),    -- contribution from NDVI stress
    feature_wind            NUMERIC(6,4),    -- contribution from wind conditions
    feature_rh              NUMERIC(6,4),    -- contribution from low humidity
    feature_slope           NUMERIC(6,4),    -- contribution from terrain slope
    feature_historical_frp  NUMERIC(6,4),   -- contribution from historical FRP

    -- Raw inputs used
    mean_ndvi_anomaly   NUMERIC(6,4),
    max_wind_ms         NUMERIC(6,2),
    min_rh_pct          NUMERIC(5,2),
    active_hotspot_count INT,
    total_frp_mw        NUMERIC(10,3),

    -- Model metadata
    model_version   TEXT         NOT NULL DEFAULT 'v1.0',
    computed_at     TIMESTAMPTZ  DEFAULT NOW(),

    CONSTRAINT uq_risk_per_day UNIQUE (pref_code, score_date)
);

CREATE INDEX IF NOT EXISTS idx_risk_score_date
    ON fire.prefecture_risk_scores (score_date DESC);
CREATE INDEX IF NOT EXISTS idx_risk_pref_date
    ON fire.prefecture_risk_scores (pref_code, score_date DESC);
CREATE INDEX IF NOT EXISTS idx_risk_level
    ON fire.prefecture_risk_scores (risk_level, score_date DESC);

-- =============================================================
-- 6. DATA INGESTION LOG
--    Tracks every pipeline run — critical for debugging & monitoring
-- =============================================================
CREATE TABLE IF NOT EXISTS fire.ingestion_log (
    id              BIGSERIAL    PRIMARY KEY,
    batch_id        UUID         NOT NULL DEFAULT gen_random_uuid(),
    source          TEXT         NOT NULL,   -- 'FIRMS_VIIRS', 'FIRMS_MODIS', 'JAXA', 'JMA', etc.
    pipeline        TEXT         NOT NULL,   -- python module that ran
    started_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    completed_at    TIMESTAMPTZ,
    status          TEXT         NOT NULL DEFAULT 'running'
                        CHECK (status IN ('running','success','partial','failed')),

    -- Coverage
    records_fetched INT          DEFAULT 0,
    records_inserted INT         DEFAULT 0,
    records_skipped  INT         DEFAULT 0,  -- duplicates

    -- Data time range that was fetched
    data_start      TIMESTAMPTZ,
    data_end        TIMESTAMPTZ,

    -- Japan bounding box used for query
    bbox            TEXT         DEFAULT '122.9,24.0,153.9,45.5',

    error_message   TEXT,
    notes           TEXT
);

CREATE INDEX IF NOT EXISTS idx_ingest_log_source
    ON fire.ingestion_log (source, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_ingest_log_status
    ON fire.ingestion_log (status, started_at DESC);

-- =============================================================
-- 7. USEFUL VIEWS
-- =============================================================

-- Latest hotspots (last 48 hours) — used by the map API directly
CREATE OR REPLACE VIEW fire.v_recent_hotspots AS
SELECT
    h.id,
    h.latitude,
    h.longitude,
    h.geom,
    h.acq_datetime,
    h.source,
    h.frp_mw,
    h.confidence,
    h.confidence_pct,
    h.daynight,
    h.pref_code,
    h.pref_name_en,
    EXTRACT(EPOCH FROM (NOW() - h.acq_datetime))/3600 AS hours_ago
FROM fire.fire_hotspots h
WHERE h.acq_datetime >= NOW() - INTERVAL '48 hours'
ORDER BY h.acq_datetime DESC;

-- Daily fire summary per prefecture — drives the risk choropleth
CREATE OR REPLACE VIEW fire.v_daily_prefecture_summary AS
SELECT
    p.pref_code,
    p.pref_name_en,
    p.pref_name_ja,
    p.geom,
    COALESCE(r.risk_score, 0)         AS risk_score,
    COALESCE(r.risk_level, 'very_low') AS risk_level,
    COALESCE(counts.hotspot_count, 0)  AS hotspot_count_24h,
    COALESCE(counts.total_frp, 0)      AS total_frp_mw_24h,
    r.score_date
FROM fire.prefecture_boundaries p
LEFT JOIN fire.prefecture_risk_scores r
    ON r.pref_code = p.pref_code
    AND r.score_date = CURRENT_DATE
LEFT JOIN (
    SELECT
        pref_code,
        COUNT(*)         AS hotspot_count,
        SUM(frp_mw)      AS total_frp
    FROM fire.fire_hotspots
    WHERE acq_datetime >= NOW() - INTERVAL '24 hours'
    GROUP BY pref_code
) counts ON counts.pref_code = p.pref_code;

-- Active fire events (cluster nearby hotspots into events)
-- Uses PostGIS ST_ClusterDBSCAN — 10km radius, minimum 1 point
CREATE OR REPLACE VIEW fire.v_fire_clusters AS
SELECT
    cluster_id,
    COUNT(*)                            AS hotspot_count,
    MIN(acq_datetime)                   AS first_detected,
    MAX(acq_datetime)                   AS last_detected,
    AVG(latitude)::NUMERIC(8,5)        AS center_lat,
    AVG(longitude)::NUMERIC(8,5)       AS center_lon,
    ST_Centroid(ST_Collect(geom))      AS center_geom,
    ST_ConvexHull(ST_Collect(geom))    AS extent_geom,
    SUM(frp_mw)                         AS total_frp_mw,
    MAX(frp_mw)                         AS max_frp_mw,
    MAX(pref_name_en)                   AS pref_name_en
FROM (
    SELECT
        *,
        ST_ClusterDBSCAN(geom, eps := 0.1, minpoints := 1)
            OVER (PARTITION BY acq_date) AS cluster_id
    FROM fire.fire_hotspots
    WHERE acq_datetime >= NOW() - INTERVAL '24 hours'
) clustered
WHERE cluster_id IS NOT NULL
GROUP BY cluster_id;

-- =============================================================
-- 8. HELPER FUNCTIONS
-- =============================================================

-- Returns GeoJSON FeatureCollection for recent hotspots (used by API)
CREATE OR REPLACE FUNCTION fire.get_hotspots_geojson(
    hours_back  INT DEFAULT 48,
    bbox_wkt    TEXT DEFAULT NULL   -- optional: 'POLYGON((...))'
)
RETURNS JSON
LANGUAGE SQL STABLE AS $$
    SELECT json_build_object(
        'type',     'FeatureCollection',
        'count',    COUNT(*),
        'generated_at', NOW(),
        'features', COALESCE(
            json_agg(
                json_build_object(
                    'type', 'Feature',
                    'geometry', ST_AsGeoJSON(h.geom)::json,
                    'properties', json_build_object(
                        'id',           h.id,
                        'source',       h.source,
                        'acq_datetime', h.acq_datetime,
                        'frp_mw',       h.frp_mw,
                        'confidence',   h.confidence,
                        'daynight',     h.daynight,
                        'pref_code',    h.pref_code,
                        'pref_name',    h.pref_name_en,
                        'hours_ago',    ROUND(EXTRACT(EPOCH FROM (NOW() - h.acq_datetime))/3600, 1)
                    )
                )
                ORDER BY h.acq_datetime DESC
            ),
            '[]'::json
        )
    )
    FROM fire.fire_hotspots h
    WHERE
        h.acq_datetime >= NOW() - (hours_back || ' hours')::INTERVAL
        AND (
            bbox_wkt IS NULL
            OR ST_Within(h.geom, ST_GeomFromText(bbox_wkt, 4326))
        );
$$;

-- Returns prefecture risk GeoJSON (choropleth layer)
CREATE OR REPLACE FUNCTION fire.get_risk_geojson(target_date DATE DEFAULT CURRENT_DATE)
RETURNS JSON
LANGUAGE SQL STABLE AS $$
    SELECT json_build_object(
        'type', 'FeatureCollection',
        'date', target_date,
        'features', COALESCE(
            json_agg(
                json_build_object(
                    'type', 'Feature',
                    'geometry', ST_AsGeoJSON(p.geom)::json,
                    'properties', json_build_object(
                        'pref_code',        p.pref_code,
                        'pref_name_en',     p.pref_name_en,
                        'pref_name_ja',     p.pref_name_ja,
                        'risk_score',       COALESCE(r.risk_score, 0),
                        'risk_level',       COALESCE(r.risk_level, 'very_low'),
                        'hotspot_count_24h',COALESCE(c.hotspot_count, 0),
                        'total_frp_mw',     COALESCE(c.total_frp, 0)
                    )
                )
            ),
            '[]'::json
        )
    )
    FROM fire.prefecture_boundaries p
    LEFT JOIN fire.prefecture_risk_scores r
        ON r.pref_code = p.pref_code AND r.score_date = target_date
    LEFT JOIN (
        SELECT pref_code, COUNT(*) hotspot_count, SUM(frp_mw) total_frp
        FROM fire.fire_hotspots
        WHERE acq_datetime >= NOW() - INTERVAL '24 hours'
        GROUP BY pref_code
    ) c ON c.pref_code = p.pref_code;
$$;

-- =============================================================
-- 9. PERMISSIONS (adjust role names to your setup)
-- =============================================================
-- Uncomment and set your actual role names:
-- CREATE ROLE fire_api_user WITH LOGIN PASSWORD 'changeme';
-- GRANT USAGE ON SCHEMA fire TO fire_api_user;
-- GRANT SELECT ON ALL TABLES IN SCHEMA fire TO fire_api_user;
-- GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA fire TO fire_api_user;
-- GRANT INSERT, UPDATE ON fire.fire_hotspots TO fire_api_user;
-- GRANT INSERT, UPDATE ON fire.weather_observations TO fire_api_user;
-- GRANT INSERT, UPDATE ON fire.prefecture_risk_scores TO fire_api_user;
-- GRANT INSERT ON fire.ingestion_log TO fire_api_user;

-- =============================================================
-- Done. Verify with:
--   \dn          (list schemas)
--   \dt fire.*   (list tables)
--   SELECT PostGIS_version();
-- =============================================================
