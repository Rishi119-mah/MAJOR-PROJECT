"""
Data loading and validation utilities for the Fake Review Detection project.

All loaders return pandas DataFrames and perform basic sanity checks.
Paths are resolved via ``src.config``.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from src.config import (
    DATA_TEXT,
    DATA_UCSD,
    DATA_WITH_IMG,
    DATA_WITHOUT_IMG,
    ID_COL,
    TARGET,
    REVIEW_FEATURES,
    NETWORK_FEATURES,
    IMAGE_FEATURES,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Loaders
# ──────────────────────────────────────────────
def load_product_data(with_images: bool = True) -> pd.DataFrame:
    """Load the product-level structured feature dataset.

    Parameters
    ----------
    with_images : bool, default True
        If True, load the 3,327-row dataset that includes image
        similarity features.  If False, load the 3,408-row superset
        that lacks image columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with ``fake`` target and ``product_ID`` identifier.
    """
    path = DATA_WITH_IMG if with_images else DATA_WITHOUT_IMG
    logger.info("Loading product data from %s", path.name)
    df = pd.read_csv(path)
    df[TARGET] = df[TARGET].astype(int)
    logger.info(
        "Loaded %d rows, %d columns.  Positive-class rate: %.2f%%",
        len(df),
        len(df.columns),
        df[TARGET].mean() * 100,
    )
    return df


def load_text_data() -> pd.DataFrame:
    """Load the TF-IDF text feature dataset (1,000 text + fake + product_ID).

    Returns
    -------
    pd.DataFrame
        3,327 rows × 1,002 columns.
    """
    logger.info("Loading text features from %s", DATA_TEXT.name)
    df = pd.read_csv(DATA_TEXT)
    df[TARGET] = df[TARGET].astype(int)
    logger.info("Loaded text data: %d rows, %d columns.", len(df), len(df.columns))
    return df


def load_ucsd_data() -> pd.DataFrame:
    """Load the UCSD unlabelled product-level dataset.

    Returns
    -------
    pd.DataFrame
        64,585 rows × 21 columns (no ``fake`` label).
    """
    logger.info("Loading UCSD data from %s", DATA_UCSD.name)
    df = pd.read_csv(DATA_UCSD)
    logger.info("Loaded UCSD data: %d rows, %d columns.", len(df), len(df.columns))
    return df


# ──────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────
def validate_data(
    df: pd.DataFrame,
    *,
    expected_target: str = TARGET,
    expected_id: str = ID_COL,
    required_features: list[str] | None = None,
) -> dict[str, Any]:
    """Run sanity checks and return a report dict.

    Checks performed:
      * Target column exists and is binary (0/1).
      * ID column exists.
      * No fully-null columns.
      * No duplicate product IDs.
      * Required features present (if supplied).

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to validate.
    expected_target : str
        Name of the target column.
    expected_id : str
        Name of the ID column.
    required_features : list[str] | None
        If given, assert every feature name is a column in *df*.

    Returns
    -------
    dict
        ``{"passed": bool, "issues": list[str]}``
    """
    issues: list[str] = []

    # Target
    if expected_target not in df.columns:
        issues.append(f"Target column '{expected_target}' not found.")
    else:
        unique_vals = set(df[expected_target].dropna().unique())
        if not unique_vals.issubset({0, 1}):
            issues.append(
                f"Target has unexpected values: {unique_vals - {0, 1}}"
            )

    # ID
    if expected_id not in df.columns:
        issues.append(f"ID column '{expected_id}' not found.")
    else:
        n_dup = df[expected_id].duplicated().sum()
        if n_dup > 0:
            issues.append(f"{n_dup} duplicate product IDs found.")

    # Null columns
    null_cols = [c for c in df.columns if df[c].isna().all()]
    if null_cols:
        issues.append(f"Fully-null columns: {null_cols}")

    # Missing values (warn, don't fail)
    n_missing = df.isna().sum().sum()
    if n_missing > 0:
        issues.append(f"{n_missing} total missing values across all columns.")

    # Required features
    if required_features is not None:
        missing_feats = [f for f in required_features if f not in df.columns]
        if missing_feats:
            issues.append(f"Missing required features: {missing_feats}")

    passed = len(issues) == 0
    if passed:
        logger.info("Data validation PASSED.")
    else:
        for iss in issues:
            logger.warning("Validation issue: %s", iss)

    return {"passed": passed, "issues": issues}


def get_dataset_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Return a concise statistical summary of a dataset.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    dict
        Keys: n_rows, n_cols, n_missing, target_distribution (if ``fake``
        column exists), numeric_stats (DataFrame.describe() as dict),
        feature_groups_present.
    """
    summary: dict[str, Any] = {
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "n_missing": int(df.isna().sum().sum()),
        "columns": list(df.columns),
    }

    if TARGET in df.columns:
        vc = df[TARGET].value_counts().to_dict()
        summary["target_distribution"] = vc
        summary["positive_rate"] = float(df[TARGET].mean())

    # Which feature groups are present?
    groups_present: dict[str, bool] = {}
    for name, feats in [
        ("review", REVIEW_FEATURES),
        ("network", NETWORK_FEATURES),
        ("image", IMAGE_FEATURES),
    ]:
        groups_present[name] = all(f in df.columns for f in feats)
    summary["feature_groups_present"] = groups_present

    # Basic numeric stats
    summary["numeric_stats"] = (
        df.select_dtypes("number").describe().to_dict()
    )

    return summary
