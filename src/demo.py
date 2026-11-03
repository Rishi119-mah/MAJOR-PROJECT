"""
Standalone Demonstration Workflow for Fake Review Detection.

Demonstrates the complete end-to-end ML & Graph analysis pipeline:
1. Data Initialization (Deterministic synthetic catalog without external 1 GB dataset)
2. Feature Validation (Model-specific schema validation)
3. Model Training / Loading (StandardScaler + Calibrated Classifier)
4. Inference (Predicting probability of fake-review manipulation)
5. Continuous Risk Scoring & Tiering (< 30 LOW, 30-64 MODERATE, >= 65 HIGH)
6. Trust-Adjusted Star Rating (Bayesian Shrinkage Heuristic)
7. Multilingual Explainability (English, Hindi, Hinglish)
8. Production FastAPI Response Simulation

Usage:
    python src/demo.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Ensure project root is in path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config import (
    ALL_STRUCTURED_FEATURES,
    NETWORK_FEATURES,
    REVIEW_FEATURES,
    REVIEW_NETWORK_FEATURES,
)
from src.evaluation.explainability import explain_single_product
from src.evaluation.risk_score import (
    compute_risk_score,
    compute_trust_adjusted_rating,
    evaluate_calibration_methods,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("demo_pipeline")


def generate_synthetic_catalog(n_samples: int = 100, seed: int = 42) -> pd.DataFrame:
    """Generate deterministic synthetic product catalog with realistic features."""
    rng = np.random.RandomState(seed)
    product_ids = [1000 + i for i in range(n_samples)]
    labels = rng.binomial(1, 0.42, size=n_samples)

    data: Dict[str, Any] = {
        "product_ID": product_ids,
        "fake": labels,
    }

    # Review Behavioral Features (12)
    data["tfidf_review_body"] = rng.uniform(0.1, 0.9, size=n_samples)
    data["n_of_reviews"] = rng.exponential(scale=50, size=n_samples) + 5
    data["avg_review_rating"] = np.where(
        labels == 1,
        rng.uniform(4.3, 5.0, size=n_samples),
        rng.uniform(2.5, 4.5, size=n_samples),
    )
    data["avg_days_between_reviews"] = rng.exponential(scale=3.0, size=n_samples) + 0.1
    data["stdev_days_between_reviews"] = rng.uniform(0.5, 10.0, size=n_samples)
    data["max_days_between_reviews"] = data["avg_days_between_reviews"] * rng.uniform(2.0, 5.0, size=n_samples)
    data["min_days_between_reviews"] = rng.uniform(0.0, 0.5, size=n_samples)
    data["share_helpful_reviews"] = rng.uniform(0.0, 0.8, size=n_samples)
    data["share_1star"] = np.where(labels == 1, rng.uniform(0.0, 0.05, size=n_samples), rng.uniform(0.05, 0.35, size=n_samples))
    data["share_5star"] = np.where(labels == 1, rng.uniform(0.70, 0.99, size=n_samples), rng.uniform(0.20, 0.65, size=n_samples))
    data["share_photo"] = rng.uniform(0.0, 0.4, size=n_samples)
    data["std_review_len"] = rng.uniform(10.0, 120.0, size=n_samples)

    # Network Centrality Features (4)
    data["pagerank"] = np.where(labels == 1, rng.uniform(0.0004, 0.002, size=n_samples), rng.uniform(0.00005, 0.0003, size=n_samples))
    data["w_degree"] = np.where(labels == 1, rng.uniform(20.0, 80.0, size=n_samples), rng.uniform(1.0, 15.0, size=n_samples))
    data["clustering_coef"] = np.where(labels == 1, rng.uniform(0.35, 0.85, size=n_samples), rng.uniform(0.0, 0.25, size=n_samples))
    data["eigenvector_cent"] = np.where(labels == 1, rng.uniform(0.05, 0.25, size=n_samples), rng.uniform(0.001, 0.04, size=n_samples))

    # Image Similarity Features (12)
    for col in [
        "min_sim", "max_sim", "mean_sim", "std_sim",
        "min_sim_review", "max_sim_review", "mean_sim_review", "std_sim_review",
        "min_sim_product", "max_sim_product", "mean_sim_product", "std_sim_product",
    ]:
        data[col] = rng.uniform(0.1, 0.95, size=n_samples)

    return pd.DataFrame(data)


def run_demo() -> Dict[str, Any]:
    print("=" * 80)
    print("  FAKE REVIEW DETECTION SYSTEM — STANDALONE DEMONSTRATION WORKFLOW")
    print("=" * 80)

    # 1. Dataset Generation
    print("\n[STEP 1/8] Generating Synthetic Catalog (No external 1 GB dataset required)...")
    df = generate_synthetic_catalog(n_samples=120, seed=42)
    print(f"  ✓ Generated {len(df)} products: {int(df['fake'].sum())} suspicious buyers, {int((df['fake'] == 0).sum())} genuine.")

    # 2. Schema Validation
    print("\n[STEP 2/8] Validating Feature Schema Adherence...")
    features = REVIEW_NETWORK_FEATURES
    missing = [f for f in features if f not in df.columns]
    assert not missing, f"Missing features: {missing}"
    print(f"  ✓ Validated {len(features)} core features (12 review + 4 network centrality).")

    # 3. Model Pipeline & Calibration
    print("\n[STEP 3/8] Fitting Development Model with Probability Calibration...")
    dev_mask = np.arange(len(df)) < 80
    df_dev, df_test = df[dev_mask], df[~dev_mask]

    X_dev = df_dev[features].values
    y_dev = df_dev["fake"].values
    X_test = df_test[features].values
    y_test = df_test["fake"].values

    base_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(n_estimators=30, max_depth=4, random_state=42)),
    ])

    cal_results, calibrated_model = evaluate_calibration_methods(
        base_pipeline, X_dev, y_dev, X_test, y_test, output_path=None
    )
    print(f"  ✓ Calibration Evaluated on Holdout: Best Method = {cal_results['best_method']}")

    # 4. Predict Single Product Cases
    print("\n[STEP 4/8] Predicting on Representative Products...")
    # Case A: Suspicious review-buying product
    case_a = df_test[df_test["fake"] == 1].iloc[0]
    # Case B: Organic genuine product
    case_b = df_test[df_test["fake"] == 0].iloc[0]

    demo_products = [
        ("Suspicious Manipulated Product", case_a),
        ("Genuine Organic Product", case_b),
    ]

    simulated_responses = []

    for label, row in demo_products:
        input_df = pd.DataFrame([row[features].to_dict()])[features]
        input_vals = input_df.values
        p_fake = float(calibrated_model.predict_proba(input_vals)[0, 1])

        # 5. Risk Scoring & Tier
        risk_score, risk_tier = compute_risk_score(p_fake)

        # 6. Trust-Adjusted Rating
        displayed_rating = float(row["avg_review_rating"])
        adjusted_rating = compute_trust_adjusted_rating(displayed_rating, p_fake)

        # 7. Localized Multilingual Explanation
        # Fallback explanation logic matching FastAPI
        if risk_score >= 65.0:
            exp_en = "Elevated risk detected due to high review clustering, 5-star concentration, and shared-reviewer density."
            exp_hi = "उच्च रिव्यू क्लस्टरिंग, 5-स्टार एकाग्रता और साझा-रिव्यूअर घनत्व के कारण उच्च जोखिम पाया गया।"
            exp_hing = "High risk detect hua hai kyunki review clustering aur 5-star percentage abnormally elevated hai."
        elif risk_score >= 30.0:
            exp_en = "Moderate risk: Some reviewer network overlap detected; recommend buyer discretion."
            exp_hi = "मध्यम जोखिम: कुछ रिव्यूअर नेटवर्क ओवरलैप देखा गया; खरीदार सावधानी रखें।"
            exp_hing = "Moderate risk: Thoda reviewer network overlap dekha gaya hai."
        else:
            exp_en = "Low risk: Review behaviors and network centrality align with normal organic purchasing."
            exp_hi = "कम जोखिम: रिव्यू व्यवहार और नेटवर्क केंद्रीयता सामान्य प्रामाणिक खरीद के अनुरूप हैं।"
            exp_hing = "Low risk: Review behavior bilkul natural aur genuine consumer pattern follow karta hai."

        # 8. Simulated FastAPI Output
        api_payload = {
            "product_ID": int(row["product_ID"]),
            "model_name": "review_network_random_forest",
            "p_fake": round(p_fake, 4),
            "is_fake_prediction": bool(p_fake >= 0.5),
            "risk_score": risk_score,
            "risk_tier": risk_tier,
            "displayed_rating": round(displayed_rating, 2),
            "trust_adjusted_rating": adjusted_rating,
            "explanation_english": exp_en,
            "explanation_hindi": exp_hi,
            "explanation_hinglish": exp_hing,
        }
        simulated_responses.append(api_payload)

        print(f"\n--- Scenario: {label} (Product ID #{api_payload['product_ID']}) ---")
        print(f"  Observed Star Rating   : {displayed_rating:.2f} ★")
        print(f"  Fake-Buyer Probability : {p_fake:.4f} ({p_fake * 100:.1f}%)")
        print(f"  Continuous Risk Score  : {risk_score:.1f} / 100 ({risk_tier})")
        print(f"  Trust-Adjusted Rating  : {adjusted_rating:.2f} ★")
        print(f"  Explanation (EN)       : {exp_en}")
        print(f"  Explanation (HI)       : {exp_hi}")
        print(f"  Explanation (Hinglish) : {exp_hing}")

    print("\n[STEP 8/8] Sample FastAPI JSON Response Payload:")
    print(json.dumps(simulated_responses[0], indent=2, ensure_ascii=False))

    print("\n" + "=" * 80)
    print("  DEMONSTRATION COMPLETED SUCCESSFULLY (100% Deterministic & Isolated)")
    print("=" * 80)

    return {"status": "success", "responses": simulated_responses}


if __name__ == "__main__":
    run_demo()
