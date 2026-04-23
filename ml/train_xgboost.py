"""
ml/train_xgboost.py
===================
Train XGBoost fire risk model from accumulated database observations.

Run this manually once you have enough data (>500 weather+hotspot rows).
The scheduler will automatically use the trained model via hybrid scoring.

Usage:
    python -m ml.train_xgboost

The model is saved to ml/xgboost_model.pkl and automatically picked up
by risk_model.py on the next scoring run.
"""

import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent / "xgboost_model.pkl"
METRICS_PATH = Path(__file__).parent / "model_metrics.json"


def fetch_training_data(engine) -> pd.DataFrame:
    """
    Build training dataset from accumulated fire hotspot history.
    Uses hotspot data + static features (forest cover, climatology).
    Weather features added where available.
    """
    from sqlalchemy import text

    sql = text("""
        WITH daily_fires AS (
            SELECT
                pref_code,
                pref_name_en,
                acq_date,
                EXTRACT(MONTH FROM acq_date)   AS month,
                EXTRACT(DOY FROM acq_date)      AS doy,
                EXTRACT(YEAR FROM acq_date)     AS year,
                COUNT(*)                        AS hotspot_count,
                COALESCE(SUM(frp_mw), 0)        AS total_frp,
                COALESCE(MAX(frp_mw), 0)        AS max_frp,
                COALESCE(AVG(frp_mw), 0)        AS avg_frp,
                COUNT(CASE WHEN daynight='D' THEN 1 END) AS day_fires,
                COUNT(CASE WHEN daynight='N' THEN 1 END) AS night_fires,
                COUNT(CASE WHEN source LIKE 'VIIRS%' THEN 1 END) AS viirs_count,
                COUNT(CASE WHEN source LIKE 'MODIS%' THEN 1 END) AS modis_count
            FROM fire.fire_hotspots
            WHERE pref_code IS NOT NULL
            GROUP BY pref_code, pref_name_en, acq_date
        ),
        -- next day fires (target variable)
        next_day AS (
            SELECT
                pref_code,
                acq_date,
                COUNT(*) AS next_count,
                COALESCE(SUM(frp_mw), 0) AS next_frp
            FROM fire.fire_hotspots
            WHERE pref_code IS NOT NULL
            GROUP BY pref_code, acq_date
        )
        SELECT
            d.pref_code,
            d.acq_date,
            d.month,
            d.doy,
            d.year,
            d.hotspot_count,
            d.total_frp,
            d.max_frp,
            d.avg_frp,
            d.day_fires,
            d.night_fires,
            d.viirs_count,
            d.modis_count,
            COALESCE(n.next_count, 0) AS next_day_hotspots,
            COALESCE(n.next_frp, 0)   AS next_day_frp
        FROM daily_fires d
        LEFT JOIN next_day n
            ON n.pref_code = d.pref_code
           AND n.acq_date  = d.acq_date + INTERVAL '1 day'
        ORDER BY d.pref_code, d.acq_date
    """)

    with engine.connect() as conn:
        df = pd.read_sql(sql, conn)

    log.info(f"Fetched {len(df):,} training rows across "
             f"{df['pref_code'].nunique()} prefectures "
             f"({df['acq_date'].min()} to {df['acq_date'].max()})")
    return df


def build_features(df: pd.DataFrame) -> tuple:
    """Build feature matrix and target vector."""
    from ml.forest_cover import get_forest_cover, get_climatology, get_wind_amplification

    df = df.copy()

    # Add static features
    df["forest_cover"] = df["pref_code"].apply(get_forest_cover)
    df["clim_factor"]  = df["month"].apply(lambda m: get_climatology(int(m)))
    df["wind_amp"]     = df["pref_code"].apply(get_wind_amplification)

    # Target: next day hotspot count → risk score 0-100
    df["target"] = np.clip(
        np.log1p(df["next_day_hotspots"]) / np.log1p(50) * 100,
        0, 100
    )

    feature_cols = [
        "month", "doy", "hotspot_count", "total_frp",
        "max_frp", "avg_frp", "day_fires", "night_fires",
        "viirs_count", "modis_count",
        "forest_cover", "clim_factor", "wind_amp"
    ]

    X = df[feature_cols].fillna(0).values
    y = df["target"].values

    return X, y, feature_cols


def train(engine=None) -> dict:
    """Train XGBoost model and save to disk."""
    try:
        import xgboost as xgb
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import mean_absolute_error, r2_score
    except ImportError:
        log.error("xgboost or sklearn not installed")
        return {"error": "missing dependencies"}

    if engine is None:
        from db.connection import get_engine
        engine = get_engine()

    df = fetch_training_data(engine)

    if len(df) < 100:
        log.warning(f"Only {len(df)} training rows — need at least 100. Skipping training.")
        return {"error": f"insufficient data: {len(df)} rows"}

    X, y, feature_cols = build_features(df)

    # Train XGBoost regressor
    model = xgb.XGBRegressor(
        n_estimators    = 200,
        max_depth       = 4,
        learning_rate   = 0.05,
        subsample       = 0.8,
        colsample_bytree= 0.8,
        reg_alpha       = 0.1,
        reg_lambda      = 1.0,
        random_state    = 42,
        n_jobs          = -1,
        verbosity       = 0,
    )

    # Cross-validation
    if len(df) >= 200:
        cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")
        log.info(f"CV R² scores: {cv_scores.round(3)} | mean: {cv_scores.mean():.3f}")
    else:
        cv_scores = np.array([0.0])

    # Final fit on all data
    model.fit(X, y)
    y_pred = model.predict(X)
    mae  = mean_absolute_error(y, y_pred)
    r2   = r2_score(y, y_pred)

    # SHAP feature importance
    try:
        import shap
        explainer    = shap.TreeExplainer(model)
        shap_values  = explainer.shap_values(X[:100])
        importance   = dict(zip(feature_cols, np.abs(shap_values).mean(axis=0)))
        top_features = sorted(importance.items(), key=lambda x: -x[1])[:5]
        log.info(f"Top SHAP features: {top_features}")
    except Exception:
        top_features = []

    # Save model
    model_obj = {
        "model":        model,
        "feature_cols": feature_cols,
        "trained_at":   datetime.now(timezone.utc).isoformat(),
        "n_samples":    len(df),
        "metrics":      {"mae": mae, "r2": r2, "cv_r2_mean": float(cv_scores.mean())},
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model_obj, f)

    log.info(f"Model saved to {MODEL_PATH} | MAE: {mae:.2f} | R²: {r2:.3f}")

    return {
        "n_samples":    len(df),
        "mae":          mae,
        "r2":           r2,
        "cv_r2_mean":   float(cv_scores.mean()),
        "top_features": top_features,
        "model_path":   str(MODEL_PATH),
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    from dotenv import load_dotenv
    load_dotenv()

    from db.connection import get_engine
    engine = get_engine()

    result = train(engine)
    if "error" in result:
        print(f"Training failed: {result['error']}")
    else:
        print(f"\nModel trained successfully:")
        print(f"  Samples:  {result['n_samples']}")
        print(f"  MAE:      {result['mae']:.2f}")
        print(f"  R²:       {result['r2']:.3f}")
        print(f"  CV R²:    {result['cv_r2_mean']:.3f}")
        print(f"\nTop features by SHAP importance:")
        for feat, val in result.get("top_features", []):
            print(f"  {feat:20s} {val:.3f}")
        print(f"\nRun 'python -m ml.risk_model' to see updated scores")
