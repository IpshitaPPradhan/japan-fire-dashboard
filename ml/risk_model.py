"""
ml/risk_model.py
================
Hybrid fire risk scoring model for Japan prefectures.

Phase 4A: Physics-informed rule-based model (live now)
Phase 4B: XGBoost ensemble merged with physics model (auto-activates
          when enough training data accumulates in the database)

Architecture mirrors operational NWP-ML hybrid systems used by
meteorological agencies (JMA, ECMWF) — physics provides the
physically-constrained baseline, ML learns residual corrections.

Risk score: 0-100 (continuous)
Risk level: very_low / low / moderate / high / very_high / extreme

Usage:
    python -m ml.risk_model
"""

import logging
import math
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent / "xgboost_model.pkl"
MIN_SAMPLES_FOR_ML = 500   # minimum DB rows before XGBoost activates


# ---------------------------------------------------------------------------
# Risk level thresholds
# ---------------------------------------------------------------------------
def score_to_level(score: float) -> str:
    if score < 15:  return "very_low"
    if score < 30:  return "low"
    if score < 50:  return "moderate"
    if score < 65:  return "high"
    if score < 80:  return "very_high"
    return "extreme"


# ---------------------------------------------------------------------------
# Physics model — component scores
# ---------------------------------------------------------------------------

def fwi_score(fwi: Optional[float]) -> float:
    """
    Convert FWI to 0-40 contribution.
    Canadian FWI: 0-5 low, 5-10 moderate, 10-20 high, 20-30 very high, 30+ extreme
    We rescale to 0-40 for our composite score.
    """
    if fwi is None or fwi < 0:
        return 10.0  # unknown → assume moderate baseline
    # Sigmoid-like mapping: FWI 0→0, 30→25, 60→35, 100→40
    return min(40.0, 40.0 * (1 - math.exp(-fwi / 45.0)))


def humidity_score(rh: Optional[float]) -> float:
    """
    Low humidity = high fire risk. Contributes 0-20 points.
    RH <20% is critically dry, >70% fire unlikely to spread.
    """
    if rh is None:
        return 8.0  # unknown → moderate baseline
    if rh >= 70:  return 0.0
    if rh >= 50:  return 4.0
    if rh >= 35:  return 10.0
    if rh >= 25:  return 15.0
    if rh >= 15:  return 18.0
    return 20.0  # <15% RH — critically dry


def wind_score(wind_ms: Optional[float], amplification: float = 1.0) -> float:
    """
    Wind speed contribution 0-20 points.
    Wind is the primary fire spread driver after ignition.
    """
    if wind_ms is None:
        return 5.0
    effective_wind = wind_ms * amplification
    if effective_wind < 2:   return 0.0
    if effective_wind < 5:   return 5.0
    if effective_wind < 8:   return 10.0
    if effective_wind < 12:  return 15.0
    if effective_wind < 16:  return 18.0
    return 20.0


def hotspot_score(hotspot_count: int, total_frp: float) -> float:
    """
    Active fire intensity contribution 0-25 points.
    Combines count (ignitions) and FRP (intensity).
    """
    if hotspot_count == 0:
        return 0.0
    count_component = min(12.0, hotspot_count * 0.8)
    frp_component   = min(13.0, math.log1p(total_frp) * 1.8)
    return count_component + frp_component


def forest_score(forest_cover: float) -> float:
    """
    Forest cover fuel load contribution 0-10 points.
    More forest = more burnable fuel.
    """
    return forest_cover * 10.0


def climatology_score(month: int, clim_factor: float) -> float:
    """
    Seasonal fire climatology contribution 0-5 points.
    Boosts risk during March-May (Japan peak fire season).
    """
    return clim_factor * 5.0


# ---------------------------------------------------------------------------
# Physics model — composite score
# ---------------------------------------------------------------------------

def physics_risk_score(
    fwi:           Optional[float],
    rh_pct:        Optional[float],
    wind_ms:       Optional[float],
    hotspot_count: int,
    total_frp:     float,
    forest_cover:  float,
    month:         int,
    clim_factor:   float,
    wind_amp:      float,
    pref_code:     str,
) -> dict:
    """
    Compute physics-informed fire risk score.
    Returns score (0-100), level, and dominant factor explanation.
    """
    components = {
        "fwi":         fwi_score(fwi),
        "humidity":    humidity_score(rh_pct),
        "wind":        wind_score(wind_ms, wind_amp),
        "hotspots":    hotspot_score(hotspot_count, total_frp),
        "forest":      forest_score(forest_cover),
        "climatology": climatology_score(month, clim_factor),
    }

    raw_score = sum(components.values())

    # Max possible = 40+20+20+25+10+5 = 120
    # Normalise to 0-100
    score = min(100.0, (raw_score / 120.0) * 100.0)
    score = round(score, 2)

    # Dominant factor for explanation
    dominant = max(components, key=components.get)
    explanations = {
        "fwi":         f"High Fire Weather Index: {fwi:.1f}" if (fwi is not None and fwi == fwi and not math.isnan(float(fwi))) else "Unknown FWI",
        "humidity":    f"Low humidity: {rh_pct:.0f}%" if rh_pct else "Low humidity",
        "wind":        f"Strong wind: {wind_ms:.1f} m/s" if wind_ms else "Wind",
        "hotspots":    f"Active fires: {hotspot_count} hotspots ({total_frp:.0f} MW FRP)",
        "forest":      f"High forest cover: {forest_cover*100:.0f}%",
        "climatology": f"Peak fire season (month {month})",
    }

    return {
        "score":       score,
        "level":       score_to_level(score),
        "dominant":    explanations[dominant],
        "components":  components,
        "model":       "physics_v1",
    }


# ---------------------------------------------------------------------------
# XGBoost model — load and predict if available
# ---------------------------------------------------------------------------

def load_xgboost_model():
    """Load trained XGBoost model if it exists."""
    if MODEL_PATH.exists():
        try:
            with open(MODEL_PATH, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            log.warning(f"Could not load XGBoost model: {e}")
    return None


def xgboost_risk_score(model, features: dict) -> Optional[float]:
    """Run XGBoost prediction. Returns score 0-100 or None if unavailable."""
    try:
        feature_vector = pd.DataFrame([features])
        score = float(model.predict(feature_vector)[0])
        return min(100.0, max(0.0, score))
    except Exception as e:
        log.warning(f"XGBoost prediction failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Data confidence weight for ML blend
# Grows from 0 to 0.7 as training data accumulates
# ---------------------------------------------------------------------------

def get_ml_weight(n_training_samples: int) -> float:
    """
    Blending weight for XGBoost vs physics model.
    0.0 = pure physics (< MIN_SAMPLES_FOR_ML samples)
    0.7 = 70% XGBoost + 30% physics (saturates at 5000 samples)
    
    Uses sigmoid growth curve for smooth transition.
    """
    if n_training_samples < MIN_SAMPLES_FOR_ML:
        return 0.0
    # Sigmoid: 0 at MIN_SAMPLES, 0.35 at 1000, 0.55 at 2000, 0.7 at 5000+
    x = (n_training_samples - MIN_SAMPLES_FOR_ML) / 1000.0
    weight = 0.7 * (1 - math.exp(-x * 0.8))
    return round(min(0.7, weight), 3)


# ---------------------------------------------------------------------------
# Hybrid model — merge physics + XGBoost
# ---------------------------------------------------------------------------

def hybrid_risk_score(
    physics_result: dict,
    xgb_model,
    features: dict,
    n_training_samples: int,
) -> dict:
    """
    Merge physics and XGBoost scores with data-confidence weighting.
    
    final = physics * (1 - w) + xgboost * w
    
    As data accumulates, w grows from 0 → 0.7, giving XGBoost
    increasing influence while physics always provides a floor.
    """
    ml_weight = get_ml_weight(n_training_samples)

    if ml_weight == 0.0 or xgb_model is None:
        return {**physics_result, "ml_weight": 0.0, "model": "physics_v1"}

    xgb_score = xgboost_risk_score(xgb_model, features)
    if xgb_score is None:
        return {**physics_result, "ml_weight": 0.0, "model": "physics_v1"}

    physics_score = physics_result["score"]
    blended = physics_score * (1 - ml_weight) + xgb_score * ml_weight
    blended = round(min(100.0, max(0.0, blended)), 2)

    return {
        "score":       blended,
        "level":       score_to_level(blended),
        "dominant":    physics_result["dominant"],
        "components":  physics_result["components"],
        "physics_score": physics_score,
        "xgb_score":   xgb_score,
        "ml_weight":   ml_weight,
        "model":       f"hybrid_v1 (physics×{1-ml_weight:.2f} + xgb×{ml_weight:.2f})",
    }


# ---------------------------------------------------------------------------
# Main scoring pipeline — runs per prefecture
# ---------------------------------------------------------------------------

def score_all_prefectures(engine=None) -> pd.DataFrame:
    """
    Compute risk scores for all 47 prefectures using latest data.
    Returns DataFrame ready to insert into fire.prefecture_risk_scores.
    """
    from ml.forest_cover import get_forest_cover, get_climatology, get_wind_amplification

    if engine is None:
        from db.connection import get_engine
        engine = get_engine()

    now = datetime.now(timezone.utc)
    month = now.month

    # ── Fetch latest weather per prefecture ──────────────────────────────
    weather_sql = """
        SELECT DISTINCT ON (pref_code)
               pref_code,
               AVG(fwi)          OVER w AS fwi,
               AVG(rh_pct)       OVER w AS rh_pct,
               AVG(wind_speed_ms)OVER w AS wind_ms,
               AVG(temp_c)       OVER w AS temp_c
        FROM fire.weather_observations
        WHERE obs_datetime >= NOW() - INTERVAL '24 hours'
        WINDOW w AS (PARTITION BY pref_code)
        ORDER BY pref_code, obs_datetime DESC
    """

    # Simpler aggregation query
    weather_sql = """
        SELECT
            w.pref_code,
            AVG(w.fwi)           AS fwi,
            AVG(w.rh_pct)        AS rh_pct,
            AVG(w.wind_speed_ms) AS wind_ms,
            AVG(w.temp_c)        AS temp_c
        FROM fire.weather_observations w
        INNER JOIN (
            SELECT pref_code, MAX(obs_datetime) AS latest
            FROM fire.weather_observations
            WHERE pref_code IS NOT NULL
            GROUP BY pref_code
        ) latest ON latest.pref_code = w.pref_code
               AND latest.latest = w.obs_datetime
        GROUP BY w.pref_code
    """

    # ── Fetch active hotspots per prefecture (last 48h) ───────────────────
    hotspot_sql = """
        SELECT
            pref_code,
            COUNT(*)        AS hotspot_count,
            SUM(frp_mw)     AS total_frp,
            MAX(frp_mw)     AS max_frp,
            COUNT(DISTINCT acq_date) AS active_days
        FROM fire.fire_hotspots
        WHERE acq_datetime >= NOW() - INTERVAL '48 hours'
          AND pref_code IS NOT NULL
        GROUP BY pref_code
    """

    # ── Fetch all prefectures ─────────────────────────────────────────────
    pref_sql = """
        SELECT pref_code, pref_name_en FROM fire.prefecture_boundaries
        ORDER BY pref_code
    """

    # ── Count training samples for ML weight ─────────────────────────────
    sample_sql = """
        SELECT COUNT(*) AS n FROM fire.weather_observations
    """

    from sqlalchemy import text
    with engine.connect() as conn:
        weather_df  = pd.read_sql(text(weather_sql),  conn)
        hotspot_df  = pd.read_sql(text(hotspot_sql),  conn)
        pref_df     = pd.read_sql(text(pref_sql),     conn)
        n_samples   = pd.read_sql(text(sample_sql),   conn).iloc[0]["n"]

    # Merge all data
    df = pref_df.copy()
    df = df.merge(weather_df,  on="pref_code", how="left")
    df = df.merge(hotspot_df,  on="pref_code", how="left")
    df["hotspot_count"] = df["hotspot_count"].fillna(0).astype(int)
    df["total_frp"]     = df["total_frp"].fillna(0.0)

    # Load XGBoost model if available
    xgb_model = load_xgboost_model()

    # ── Score each prefecture ─────────────────────────────────────────────
    results = []
    clim_factor = get_climatology(month)

    for _, row in df.iterrows():
        pref_code    = row["pref_code"]
        forest_cover = get_forest_cover(pref_code)
        wind_amp     = get_wind_amplification(pref_code)

        def safe(val):
            try:
                v = float(val)
                return None if math.isnan(v) else v
            except (TypeError, ValueError):
                return None

        physics = physics_risk_score(
            fwi           = safe(row.get("fwi")),
            rh_pct        = safe(row.get("rh_pct")),
            wind_ms       = safe(row.get("wind_ms")),
            hotspot_count = int(row["hotspot_count"]),
            total_frp     = float(row["total_frp"]),
            forest_cover  = forest_cover,
            month         = month,
            clim_factor   = clim_factor,
            wind_amp      = wind_amp,
            pref_code     = pref_code,
        )

        features = {
            "fwi":           safe(row.get("fwi")) or 0,
            "rh_pct":        safe(row.get("rh_pct")) or 50,
            "wind_ms":       safe(row.get("wind_ms")) or 0,
            "temp_c":        safe(row.get("temp_c")) or 15,
            "hotspot_count": int(row["hotspot_count"]),
            "total_frp":     float(row["total_frp"]),
            "forest_cover":  forest_cover,
            "month":         month,
            "clim_factor":   clim_factor,
            "wind_amp":      wind_amp,
        }

        final = hybrid_risk_score(physics, xgb_model, features, int(n_samples))

        results.append({
            "pref_code":      pref_code,
            "pref_name_en":   row["pref_name_en"],
            "score_date":     now.date(),
            "risk_score":     final["score"],
            "risk_level":     final["level"],
            "dominant_factor":final["dominant"],
            "model_version":  final["model"],
            "ml_weight":      final.get("ml_weight", 0.0),
            "physics_score":  final.get("physics_score", final["score"]),
            "xgb_score":      final.get("xgb_score"),
            "computed_at":    now,
        })

    result_df = pd.DataFrame(results)
    log.info(f"Scored {len(result_df)} prefectures | model: {results[0]['model_version']} | "
             f"ML weight: {results[0]['ml_weight']:.2f} | "
             f"training samples: {n_samples}")
    return result_df


# ---------------------------------------------------------------------------
# Write scores to database
# ---------------------------------------------------------------------------

def write_scores(df: pd.DataFrame, engine=None) -> int:
    """Upsert risk scores into fire.prefecture_risk_scores."""
    if engine is None:
        from db.connection import get_engine
        engine = get_engine()

    from sqlalchemy import text
    sql = text("""
        INSERT INTO fire.prefecture_risk_scores (
            pref_code, pref_name_en, score_date,
            risk_score, risk_level, model_version
        ) VALUES (
            :pref_code, :pref_name_en, :score_date,
            :risk_score, :risk_level, :model_version
        )
        ON CONFLICT (pref_code, score_date) DO UPDATE SET
            risk_score     = EXCLUDED.risk_score,
            risk_level     = EXCLUDED.risk_level,
            model_version  = EXCLUDED.model_version
    """)

    with engine.connect() as conn:
        for _, row in df.iterrows():
            conn.execute(sql, {
                "pref_code":    row["pref_code"],
                "pref_name_en": row["pref_name_en"],
                "score_date":   row["score_date"],
                "risk_score":   float(row["risk_score"]),
                "risk_level":   row["risk_level"],
                "model_version":row["model_version"],
            })
        conn.commit()

    log.info(f"Wrote {len(df)} risk scores to database")
    return len(df)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_risk_scoring(engine=None) -> dict:
    """Full risk scoring pipeline. Called by scheduler every hour."""
    if engine is None:
        from db.connection import get_engine
        engine = get_engine()

    try:
        df = score_all_prefectures(engine)
        n  = write_scores(df, engine)

        # Summary stats
        high_risk = df[df["risk_level"].isin(["high", "very_high", "extreme"])]
        top3 = df.nlargest(3, "risk_score")[["pref_name_en", "risk_score", "dominant_factor"]]

        return {
            "scored":     n,
            "high_risk":  len(high_risk),
            "top3":       top3.to_dict(orient="records"),
            "model":      df["model_version"].iloc[0],
            "ml_weight":  float(df["ml_weight"].iloc[0]),
        }
    except Exception as e:
        log.error(f"Risk scoring failed: {e}", exc_info=True)
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Main — run standalone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    from dotenv import load_dotenv
    load_dotenv()

    result = run_risk_scoring()
    print(f"\nRisk scoring complete:")
    print(f"  Prefectures scored: {result.get('scored')}")
    print(f"  High risk prefectures: {result.get('high_risk')}")
    print(f"  Model: {result.get('model')}")
    print(f"  ML weight: {result.get('ml_weight', 0):.2f}")
    print(f"\nTop 3 highest risk:")
    for r in result.get("top3", []):
        print(f"  {r['pref_name_en']:15s} {r['risk_score']:5.1f}  ({r['dominant_factor']})")
