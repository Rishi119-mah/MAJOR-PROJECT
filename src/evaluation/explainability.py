"""
SHAP Explainability Module for Fake Review Detection.

Implements TreeSHAP analysis for tree models (XGBoost, Random Forest).
Validates feature group importance and computes exact percentage contributions.
Generates both global summaries and localized product explanations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.config import FIGURE_DIR, OUTPUT_DIR, TARGET

logger = logging.getLogger(__name__)


def compute_shap_explanations(
    model: Any,
    X: pd.DataFrame,
    feature_names: list[str],
    max_display: int = 15,
) -> dict[str, Any]:
    """Compute TreeSHAP values and global feature importances.

    Parameters
    ----------
    model : fitted tree estimator (e.g., XGBClassifier or RandomForestClassifier)
    X : pd.DataFrame
        Background / evaluation feature matrix.
    feature_names : list[str]
        List of feature column names.
    max_display : int
        Number of top features to plot.

    Returns
    -------
    dict[str, Any]
        Dictionary with shap_values, mean_abs_shap, percent_importance, and summary dict.
    """
    logger.info("Computing SHAP values via TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # In binary classification, some models return 2D array, some return list of 2 arrays
    if isinstance(shap_values, list):
        # Index 1 corresponds to positive class (fake=1)
        sv = shap_values[1]
    elif len(shap_values.shape) == 3:
        sv = shap_values[:, :, 1]
    else:
        sv = shap_values

    # Mean absolute SHAP values per feature
    mean_abs = np.abs(sv).mean(axis=0)
    total_abs = mean_abs.sum()
    percent_importance = (mean_abs / total_abs) * 100.0 if total_abs > 0 else np.zeros_like(mean_abs)

    feature_importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs,
        "percent_importance": percent_importance,
    }).sort_values(by="mean_abs_shap", ascending=False)

    # Save summary plot
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(sv, X, feature_names=feature_names, show=False, max_display=max_display)
    plot_path = FIGURE_DIR / "shap_summary_plot.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved SHAP summary plot to %s", plot_path)

    # Save importance table
    csv_path = OUTPUT_DIR / "experiments" / "SHAP_FEATURE_IMPORTANCE.csv"
    feature_importance_df.to_csv(csv_path, index=False)
    logger.info("Saved SHAP feature importances to %s", csv_path)

    return {
        "explainer": explainer,
        "shap_values": sv,
        "feature_importance_df": feature_importance_df,
        "plot_path": str(plot_path),
    }


def explain_single_product(
    explainer: Any,
    product_features: pd.DataFrame,
    feature_names: list[str],
    top_k: int = 5,
) -> dict[str, Any]:
    """Generate localized, human-interpretable explanations for an individual product.

    Returns top risk-increasing and risk-reducing factors in English, Hindi, and Hinglish.
    """
    sv = explainer.shap_values(product_features)
    if isinstance(sv, list):
        sv = sv[1]
    elif len(sv.shape) == 3:
        sv = sv[:, :, 1]
    elif len(sv.shape) == 2 and sv.shape[0] == 1:
        sv = sv[0]

    vals = sv.flatten() if hasattr(sv, "flatten") else sv
    feat_contribs = list(zip(feature_names, vals, product_features.values.flatten()))
    feat_contribs.sort(key=lambda x: abs(x[1]), reverse=True)

    risk_increasing = [f for f in feat_contribs if f[1] > 0][:top_k]
    risk_decreasing = [f for f in feat_contribs if f[1] < 0][:top_k]

    # Human-readable feature name mapping
    friendly_names = {
        "w_degree": ("Co-review network degree (connections to shared reviewers)", "नेटवर्क को-रिव्यू डिग्री"),
        "clustering_coef": ("Reviewer network clustering coefficient", "रिव्यूअर नेटवर्क क्लस्टरिंग गुणांक"),
        "eigenvector_cent": ("Eigenvector centrality in buyer network", "खरीदार नेटवर्क में केंद्रीयता"),
        "pagerank": ("PageRank influence in reviewer graph", "पेजरैंक प्रभाव"),
        "share_5star": ("Proportion of 5-star ratings", "5-स्टार रेटिंग का अनुपात"),
        "avg_days_between_reviews": ("Average days between consecutive reviews", "रिव्यू के बीच औसत दिन"),
        "share_photo": ("Proportion of photo reviews", "फोटो वाले रिव्यू का अनुपात"),
    }

    # Generate explanations
    en_factors = [f"{friendly_names.get(f[0], (f[0], ''))[0]} (+{f[1]:.3f})" for f in risk_increasing]
    hi_factors = [f"{friendly_names.get(f[0], ('', f[0]))[1]} (+{f[1]:.3f})" for f in risk_increasing]

    return {
        "top_risk_factors": risk_increasing,
        "top_protective_factors": risk_decreasing,
        "english_explanation": (
            f"Primary factors elevating risk: {'; '.join(en_factors)}."
            if en_factors else "Features are consistent with normal genuine product behavior."
        ),
        "hindi_explanation": (
            f"जोखिम बढ़ाने वाले मुख्य कारक: {'; '.join(hi_factors)}."
            if hi_factors else "उत्पाद की विशेषताएं सामान्य और प्रामाणिक पाई गई हैं।"
        ),
        "hinglish_explanation": (
            f"Risk badhane wale main factors: {'; '.join(en_factors)}. Inka unusual pattern coordinated review buying indicate karta hai."
            if en_factors else "Product ke sabhi features genuine aur normal review patterns ke anuroop hain."
        ),
    }

