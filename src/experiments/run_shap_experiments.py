"""
Run SHAP analysis on the trained XGBoost model.
Computes:
1. Exact global feature importance (mean absolute SHAP and percentage contribution).
2. Investigates whether w_degree or network features account for ~21% or reports exact percentage.
3. Generates summary plots.
4. Generates local explanation samples.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import ALL_STRUCTURED_FEATURES, FIGURE_DIR, MODEL_DIR, TARGET
from src.data.data_loader import load_product_data
from src.evaluation.explainability import compute_shap_explanations, explain_single_product

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("shap_experiment")


def main():
    logger.info("Loading product dataset...")
    df = load_product_data(with_images=True)
    X = df[ALL_STRUCTURED_FEATURES]
    y = df[TARGET]

    # Load trained XGBoost model from model directory
    model_path = MODEL_DIR / "review_network_image_xgboost_pipeline.joblib"
    if not model_path.exists():
        logger.warning("Pipeline %s not found. Training a fresh XGBoost model...", model_path)
        import xgboost as xgb
        model = xgb.XGBClassifier(eval_metric="logloss", random_state=42)
        model.fit(X, y)
    else:
        pipeline = joblib.load(model_path)
        model = pipeline.named_steps["classifier"] if hasattr(pipeline, "named_steps") else pipeline

    # Run SHAP on background sample (e.g. 1000 samples for fast, highly accurate estimation)
    sample_X = X.sample(n=min(1000, len(X)), random_state=42)
    res = compute_shap_explanations(model, sample_X, ALL_STRUCTURED_FEATURES)

    imp_df = res["feature_importance_df"]
    print("\n" + "=" * 60)
    print("SHAP GLOBAL FEATURE IMPORTANCES (TOP 15):")
    print("=" * 60)
    print(imp_df.head(15).to_string(index=False))

    # Check network feature group total percentage
    network_feats = ["w_degree", "pagerank", "clustering_coef", "eigenvector_cent"]
    net_imp = imp_df[imp_df["feature"].isin(network_feats)]["percent_importance"].sum()
    print("\n" + "=" * 60)
    print(f"TOTAL NETWORK FEATURE IMPORTANCE: {net_imp:.2f}%")
    w_deg_imp = imp_df[imp_df["feature"] == "w_degree"]["percent_importance"].values
    if len(w_deg_imp) > 0:
        print(f"w_degree IMPORTANCE: {w_deg_imp[0]:.2f}% (compare to PPT claim of ~21%)")
    print("=" * 60)

    # Explain 2 sample products: 1 fake product, 1 genuine product
    fake_sample = X[y == 1].iloc[0:1]
    gen_sample = X[y == 0].iloc[0:1]

    print("\nEXPLANATION FOR A DETECTED FAKE-REVIEW BUYER:")
    exp_fake = explain_single_product(res["explainer"], fake_sample, ALL_STRUCTURED_FEATURES)
    print("English:", exp_fake["english_explanation"])
    print("Hindi  :", exp_fake["hindi_explanation"])
    print("Hinglish:", exp_fake["hinglish_explanation"])

    print("\nEXPLANATION FOR A GENUINE PRODUCT:")
    exp_gen = explain_single_product(res["explainer"], gen_sample, ALL_STRUCTURED_FEATURES)
    print("English:", exp_gen["english_explanation"])
    print("Hindi  :", exp_gen["hindi_explanation"])


if __name__ == "__main__":
    main()

