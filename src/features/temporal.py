"""
Temporal Dynamics Feature Extraction and Analysis Module.

Extracts burstiness, velocity, rating entropy, and temporal concentration metrics
from product-level inter-review statistics.

DATA AVAILABILITY NOTE:
Raw review-level timestamps reside in the ~1 GB 'UCSD_home_and_kitchen_reviews.csv.gz'
dataset (not provided in the local repository). To maintain scientific integrity without
fabricating artificial timestamps, temporal analysis rigorously leverages the verified
inter-review interval features already present in the product dataset:
- avg_days_between_reviews
- stdev_days_between_reviews
- max_days_between_reviews
- min_days_between_reviews
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from src.config import EXPERIMENT_DIR, RANDOM_SEED, REVIEW_NETWORK_FEATURES, TARGET
from src.evaluation.metrics import compute_all_metrics

logger = logging.getLogger(__name__)

TEMPORAL_FEATURE_NAMES = [
    "review_velocity",
    "review_burstiness",
    "rating_entropy",
    "temporal_concentration",
]


def compute_temporal_proxy_features(df_product: pd.DataFrame) -> pd.DataFrame:
    """Compute temporal dynamics features from product-level inter-review statistics.

    Features generated:
      - review_velocity: Rate of reviews accumulated per day over the product lifespan.
      - review_burstiness: Coefficient of variation (sigma / mu) of inter-review intervals.
      - rating_entropy: Shannon entropy across the bimodal rating distribution.
      - temporal_concentration: Inverse of average inter-review days.
    """
    df = df_product.copy()

    # 1. Review Velocity: reviews per day
    total_span_days = df["avg_days_between_reviews"] * (df["n_of_reviews"] - 1).clip(lower=1)
    df["review_velocity"] = df["n_of_reviews"] / total_span_days.clip(lower=1.0)

    # 2. Review Burstiness: coefficient of variation of inter-review intervals
    df["review_burstiness"] = df["stdev_days_between_reviews"] / df["avg_days_between_reviews"].clip(lower=0.01)

    # 3. Rating Entropy: Shannon entropy across 1-star, 5-star, and middle ratings
    def calc_entropy(row: pd.Series) -> float:
        p1 = max(0.0, float(row["share_1star"]))
        p5 = max(0.0, float(row["share_5star"]))
        pmid = max(0.0, 1.0 - (p1 + p5))
        probs = [p1, pmid, p5]
        probs = [p for p in probs if p > 0.0]
        if not probs:
            return 0.0
        total = sum(probs)
        probs = [p / total for p in probs]
        return float(entropy(probs, base=2))

    df["rating_entropy"] = df.apply(calc_entropy, axis=1)

    # 4. Temporal Concentration: inverse inter-review time
    df["temporal_concentration"] = 1.0 / (df["avg_days_between_reviews"].clip(lower=0.1))

    logger.info("Computed 4 temporal features: %s", TEMPORAL_FEATURE_NAMES)
    return df


def evaluate_temporal_feature_impact(
    df_product: pd.DataFrame,
) -> Dict[str, Any]:
    """Empirically evaluate the additive value of temporal features over Review+Network features."""
    df_with_temp = compute_temporal_proxy_features(df_product)

    # Base features: Review + Network
    X_base = df_with_temp[REVIEW_NETWORK_FEATURES].values
    # Base + Temporal
    X_temp = df_with_temp[REVIEW_NETWORK_FEATURES + TEMPORAL_FEATURE_NAMES].values
    y = df_with_temp[TARGET].values

    # Stratified 80/20 split
    X_tr_b, X_te_b, y_tr, y_te = train_test_split(X_base, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y)
    X_tr_t, X_te_t, _, _ = train_test_split(X_temp, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y)

    scaler_b = StandardScaler()
    X_tr_b = scaler_b.fit_transform(X_tr_b)
    X_te_b = scaler_b.transform(X_te_b)

    scaler_t = StandardScaler()
    X_tr_t = scaler_t.fit_transform(X_tr_t)
    X_te_t = scaler_t.transform(X_te_t)

    # Fit XGBoost on Base
    clf_base = xgb.XGBClassifier(eval_metric="logloss", random_state=RANDOM_SEED, n_estimators=100, max_depth=5)
    clf_base.fit(X_tr_b, y_tr)
    pred_b = clf_base.predict(X_te_b)
    prob_b = clf_base.predict_proba(X_te_b)[:, 1]
    metrics_base = compute_all_metrics(y_te, pred_b, prob_b)

    # Fit XGBoost on Base + Temporal
    clf_temp = xgb.XGBClassifier(eval_metric="logloss", random_state=RANDOM_SEED, n_estimators=100, max_depth=5)
    clf_temp.fit(X_tr_t, y_tr)
    pred_t = clf_temp.predict(X_te_t)
    prob_t = clf_temp.predict_proba(X_te_t)[:, 1]
    metrics_temp = compute_all_metrics(y_te, pred_t, prob_t)

    comparison = [
        {"feature_set": "Review + Network (16 features)", **metrics_base},
        {"feature_set": "Review + Network + Temporal (20 features)", **metrics_temp},
    ]

    df_comp = pd.DataFrame(comparison)
    out_csv = EXPERIMENT_DIR / "TEMPORAL_FEATURE_EVALUATION.csv"
    df_comp.to_csv(out_csv, index=False)
    logger.info("Saved temporal evaluation to %s", out_csv)

    return {"comparison_df": df_comp, "metrics_base": metrics_base, "metrics_temp": metrics_temp}
