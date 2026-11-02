"""
Temporal feature extraction and chronological validation splits.

Extracts burstiness, velocity, rating entropy, and temporal concentration metrics
from review-level timestamp data, or computes product-level proxies from summary stats.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import entropy

logger = logging.getLogger(__name__)


def compute_temporal_proxy_features(df_product: pd.DataFrame) -> pd.DataFrame:
    """Compute temporal dynamics features from product-level summary statistics.

    Features generated:
      - review_velocity: n_of_reviews / max(1.0, max_days_between_reviews * n_of_reviews)
      - review_burstiness: stdev_days_between_reviews / max(1.0, avg_days_between_reviews)
      - rating_entropy: Shannon entropy across star rating distribution
      - temporal_concentration: min_days_between_reviews / max(1.0, avg_days_between_reviews)
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
        # Normalize
        total = sum(probs)
        probs = [p / total for p in probs]
        return float(entropy(probs, base=2))

    df["rating_entropy"] = df.apply(calc_entropy, axis=1)

    # 4. Temporal Concentration
    df["temporal_concentration"] = 1.0 / (df["avg_days_between_reviews"].clip(lower=0.1))

    logger.info("Computed 4 temporal features: review_velocity, review_burstiness, rating_entropy, temporal_concentration")
    return df

