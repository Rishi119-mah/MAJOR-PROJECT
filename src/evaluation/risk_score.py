"""
Product Risk Indicator and Trust-Adjusted Rating Module.

Translates model probabilities into consumer-facing risk metrics:
- True Probability Calibration (Platt Sigmoid and Isotonic Regression via CalibratedClassifierCV)
- Expected Calibration Error (ECE) & Brier Score Diagnostics
- 0-100 Continuous Risk Score with Tested Boundaries (< 30 LOW, 30-64 MODERATE, >= 65 HIGH)
- Trust-Adjusted Product Star Rating (Bayesian Shrinkage Heuristic towards Reference Rating)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple, Optional, Union
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score, f1_score
from sklearn.base import BaseEstimator, clone

from src.config import EXPERIMENT_DIR, RANDOM_SEED

logger = logging.getLogger(__name__)


def compute_risk_score(
    p_fake: float,
) -> Tuple[float, str]:
    """Convert a fake-class probability into a 0-100 continuous risk score and tier.

    Strict Risk Band Boundaries:
    - [0.0, 30.0): "LOW_RISK"
    - [30.0, 65.0): "MODERATE_RISK"
    - [65.0, 100.0]: "HIGH_RISK"
    """
    p_clipped = float(np.clip(p_fake, 0.0, 1.0))
    risk_score = round(p_clipped * 100.0, 2)

    if risk_score < 30.0:
        risk_tier = "LOW_RISK"
    elif risk_score < 65.0:
        risk_tier = "MODERATE_RISK"
    else:
        risk_tier = "HIGH_RISK"

    return risk_score, risk_tier


def compute_trust_adjusted_rating(
    displayed_rating: float,
    p_fake: float,
    reference_rating: float = 3.50,
) -> float:
    """Compute trust-adjusted rating discounting manipulated star inflation.

    Formula (Bayesian Shrinkage Heuristic):
      Adjusted Rating = (1 - p_fake) * displayed_rating + p_fake * reference_rating

    Parameters
    ----------
    displayed_rating : float
        Current Amazon star rating (e.g. 4.7).
    p_fake : float
        Calibrated fake-class probability [0.0, 1.0].
    reference_rating : float, default 3.50
        Category baseline rating for unmanipulated products.
    """
    p_clipped = float(np.clip(p_fake, 0.0, 1.0))
    adjusted = (1.0 - p_clipped) * displayed_rating + p_clipped * reference_rating
    return float(np.clip(round(adjusted, 2), 1.0, 5.0))


def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE) across uniform probability bins."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    ece = 0.0
    n_total = len(y_true)

    for b in range(n_bins):
        mask = bin_indices == b
        if np.any(mask):
            prop_b = np.sum(mask) / n_total
            acc_b = np.mean(y_true[mask])
            conf_b = np.mean(y_prob[mask])
            ece += prop_b * np.abs(acc_b - conf_b)

    return float(ece)


def evaluate_calibration_methods(
    base_model: BaseEstimator,
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_path: Optional[Union[str, Path]] = None,
    val_fraction: float = 0.25,
    random_state: int = RANDOM_SEED,
) -> Tuple[Dict[str, Any], BaseEstimator]:
    """Fit and compare Raw, Platt Sigmoid, and Isotonic calibration.

    LEAKAGE-FREE CALIBRATION SELECTION GUARANTEE:
    1. Selection strictly on Development Data:
       X_dev is split into Sub-Train (75%) and Sub-Val (25%).
       Raw, Platt Sigmoid, and Isotonic methods are fitted on Sub-Train and compared
       strictly on Sub-Val Brier score. Final test data (X_test, y_test) is NEVER
       accessed during model/calibration selection.
    2. Selected Method Fitting:
       The winning calibration method from Sub-Val is fitted on the entire Development Set (X_dev, y_dev).
    3. Holdout Evaluation:
       All candidate methods and the chosen calibrated model are evaluated ONCE on X_test, y_test
       to record holdout performance.
    """
    from sklearn.model_selection import train_test_split

    # 1. Dev Sub-Validation Split for Leakage-Free Selection
    X_sub_tr, X_sub_val, y_sub_tr, y_sub_val = train_test_split(
        X_dev, y_dev, test_size=val_fraction, random_state=random_state, stratify=y_dev
    )

    # A. Raw Model on Dev Val
    raw_sub = clone(base_model).fit(X_sub_tr, y_sub_tr)
    prob_raw_val = raw_sub.predict_proba(X_sub_val)[:, 1]
    brier_val_raw = brier_score_loss(y_sub_val, prob_raw_val)

    # B. Sigmoid on Dev Val
    sig_sub = CalibratedClassifierCV(estimator=clone(base_model), method="sigmoid", cv=3)
    sig_sub.fit(X_sub_tr, y_sub_tr)
    prob_sig_val = sig_sub.predict_proba(X_sub_val)[:, 1]
    brier_val_sig = brier_score_loss(y_sub_val, prob_sig_val)

    # C. Isotonic on Dev Val
    iso_sub = CalibratedClassifierCV(estimator=clone(base_model), method="isotonic", cv=3)
    iso_sub.fit(X_sub_tr, y_sub_tr)
    prob_iso_val = iso_sub.predict_proba(X_sub_val)[:, 1]
    brier_val_iso = brier_score_loss(y_sub_val, prob_iso_val)

    val_briers = {
        "Raw (Uncalibrated)": (brier_val_raw, "none"),
        "Platt Sigmoid Calibration": (brier_val_sig, "sigmoid"),
        "Isotonic Regression Calibration": (brier_val_iso, "isotonic"),
    }
    best_name = min(val_briers, key=lambda k: val_briers[k][0])
    selected_method_key = val_briers[best_name][1]

    logger.info(
        "Calibration method selected strictly on Dev Validation: %s (Val Brier: %.4f). Test set not accessed for selection.",
        best_name, val_briers[best_name][0]
    )

    # 2. Fit Full Models on Development Data (X_dev, y_dev)
    model_raw = clone(base_model).fit(X_dev, y_dev)
    model_sig = CalibratedClassifierCV(estimator=clone(base_model), method="sigmoid", cv=5).fit(X_dev, y_dev)
    model_iso = CalibratedClassifierCV(estimator=clone(base_model), method="isotonic", cv=5).fit(X_dev, y_dev)

    if selected_method_key == "isotonic":
        best_model = model_iso
    elif selected_method_key == "sigmoid":
        best_model = model_sig
    else:
        best_model = model_raw

    # 3. Evaluate ONCE on Untouched Final Test Set (X_test, y_test)
    results = []

    # Raw Test
    prob_raw_test = model_raw.predict_proba(X_test)[:, 1]
    results.append({
        "calibration_method": "Raw (Uncalibrated)",
        "dev_val_brier_score": float(brier_val_raw),
        "selected_on_dev": best_name == "Raw (Uncalibrated)",
        "brier_score": float(brier_score_loss(y_test, prob_raw_test)),
        "expected_calibration_error": float(compute_expected_calibration_error(y_test, prob_raw_test)),
        "roc_auc": float(roc_auc_score(y_test, prob_raw_test)),
        "fake_f1": float(f1_score(y_test, (prob_raw_test >= 0.5).astype(int), pos_label=1)),
    })

    # Sigmoid Test
    prob_sig_test = model_sig.predict_proba(X_test)[:, 1]
    results.append({
        "calibration_method": "Platt Sigmoid Calibration",
        "dev_val_brier_score": float(brier_val_sig),
        "selected_on_dev": best_name == "Platt Sigmoid Calibration",
        "brier_score": float(brier_score_loss(y_test, prob_sig_test)),
        "expected_calibration_error": float(compute_expected_calibration_error(y_test, prob_sig_test)),
        "roc_auc": float(roc_auc_score(y_test, prob_sig_test)),
        "fake_f1": float(f1_score(y_test, (prob_sig_test >= 0.5).astype(int), pos_label=1)),
    })

    # Isotonic Test
    prob_iso_test = model_iso.predict_proba(X_test)[:, 1]
    results.append({
        "calibration_method": "Isotonic Regression Calibration",
        "dev_val_brier_score": float(brier_val_iso),
        "selected_on_dev": best_name == "Isotonic Regression Calibration",
        "brier_score": float(brier_score_loss(y_test, prob_iso_test)),
        "expected_calibration_error": float(compute_expected_calibration_error(y_test, prob_iso_test)),
        "roc_auc": float(roc_auc_score(y_test, prob_iso_test)),
        "fake_f1": float(f1_score(y_test, (prob_iso_test >= 0.5).astype(int), pos_label=1)),
    })

    df_cal = pd.DataFrame(results)
    if output_path is not None:
        out_csv = Path(output_path)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df_cal.to_csv(out_csv, index=False)
        logger.info("Saved probability calibration comparison to %s", out_csv)

    return {"summary_df": df_cal, "best_method": best_name, "calibration_method_key": selected_method_key}, best_model

    logger.info("Selected optimal calibrated model: %s (Brier: %.4f, ECE: %.4f)",
                best_name, best_row["brier_score"], best_row["expected_calibration_error"])

    return {"summary_df": df_cal, "best_method": best_name}, best_model
