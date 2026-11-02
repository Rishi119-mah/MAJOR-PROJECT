"""
Tests for FastAPI application endpoints and route logic.
"""

from __future__ import annotations

import sys
from pathlib import Path
import pytest

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.api.main import (
    health,
    get_datasets_summary,
    get_metrics,
    predict_product,
    get_product_details,
    get_clustering_summary,
    ProductFeatureInput,
)


def test_health():
    res = health()
    assert res["status"] == "ok"
    assert res["version"] == "1.0.0"


def test_datasets_summary():
    res = get_datasets_summary()
    assert res["labeled_products"] == 3327
    assert res["positive_class_count"] == 1400
    assert res["fake_prevalence_pct"] > 40.0


def test_models_metrics():
    res = get_metrics()
    assert isinstance(res, list)
    assert len(res) > 0
    # Check expected keys in metrics
    sample = res[0]
    assert "accuracy" in sample
    assert "fake_f1" in sample
    assert "roc_auc" in sample


def test_predict_product_direct():
    payload = ProductFeatureInput(
        product_ID=12345,
        displayed_rating=4.7,
        features={
            "w_degree": 30.0,
            "clustering_coef": 0.50,
            "eigenvector_cent": 0.08,
            "pagerank": 0.0003,
            "share_5star": 0.85,
            "share_photo": 0.25,
            "n_of_reviews": 80.0,
            "avg_review_rating": 4.7,
            "avg_days_between_reviews": 1.8,
        },
    )
    res = predict_product(payload)
    assert "p_fake" in res
    assert "risk_score" in res
    assert 0.0 <= res["risk_score"] <= 100.0
    assert res["risk_tier"] in ["LOW_RISK", "MODERATE_RISK", "HIGH_RISK"]
    assert "trust_adjusted_rating" in res
    assert 1.0 <= res["trust_adjusted_rating"] <= 5.0
    assert len(res["explanation_english"]) > 0
    assert len(res["explanation_hindi"]) > 0
    assert len(res["explanation_hinglish"]) > 0


def test_get_product_details():
    # Pick first product from labelled dataset
    from src.data.data_loader import load_product_data
    df = load_product_data(with_images=True)
    first_pid = int(df["product_ID"].iloc[0])

    res = get_product_details(product_id=first_pid)
    assert res["product_ID"] == first_pid
    assert "ground_truth_label" in res
    assert "features" in res
    assert "w_degree" in res["features"]


def test_clustering_summary():
    res = get_clustering_summary()
    assert isinstance(res, list)
    assert len(res) > 0
    assert "cluster_ID" in res[0]
