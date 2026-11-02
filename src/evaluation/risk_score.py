"""
Product Risk Indicator and Trust-Adjusted Rating Module.

Translates model probabilities into consumer-facing risk metrics:
- 0-100 Risk Score
- Risk Category Tiers
- Trust-Adjusted Product Star Rating
- Probability Calibration (Brier score & reliability diagnostics)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

logger = logging.getLogger(__name__)


def compute_risk_score(
    p_fake: float,
) -> tuple[float, str]:
    """Convert a calibrated fake-class probability into a 0-100 risk score and tier.

    Parameters
    ----------
    p_fake : float
        Calibrated probability that product buys fake reviews in [0.0, 1.0].

    Returns
    -------
    risk_score : float in [0.0, 100.0]
    risk_tier : str ("LOW_RISK", "MODERATE_RISK", "HIGH_RISK")
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

    Formula:
      Adjusted Rating = (1 - p_fake) * displayed_rating + p_fake * reference_rating

    Parameters
    ----------
    displayed_rating : float
        Current Amazon star rating (e.g. 4.7).
    p_fake : float
        Calibrated fake-class probability [0.0, 1.0].
    reference_rating : float, default 3.50
        Category baseline rating for unmanipulated products.

    Returns
    -------
    float
        Trust-adjusted star rating clipped to [1.0, 5.0].
    """
    p_clipped = float(np.clip(p_fake, 0.0, 1.0))
    adjusted = (1.0 - p_clipped) * displayed_rating + p_clipped * reference_rating
    return float(np.clip(round(adjusted, 2), 1.0, 5.0))


def evaluate_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Evaluate probability calibration quality.

    Computes Brier score loss and calibration reliability curve.
    """
    brier = float(brier_score_loss(y_true, y_prob))
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")

    # Expected Calibration Error (ECE)
    ece = float(np.mean(np.abs(prob_true - prob_pred)))

    return {
        "brier_score": brier,
        "expected_calibration_error": ece,
        "prob_true": prob_true.tolist(),
        "prob_pred": prob_pred.tolist(),
    }

