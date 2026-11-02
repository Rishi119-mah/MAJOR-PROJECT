"""
Unit and integration tests for data loading, preprocessing, metrics, and risk scoring.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pytest

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data.data_loader import load_product_data, validate_data, get_dataset_summary
from src.evaluation.metrics import compute_all_metrics
from src.evaluation.risk_score import compute_risk_score, compute_trust_adjusted_rating
from src.graph.synthetic_rings import generate_synthetic_review_ring
from src.graph.graph_builder import build_projected_product_graph, extract_graph_features


def test_data_loading_and_validation():
    df = load_product_data(with_images=True)
    assert len(df) == 3327
    assert "fake" in df.columns
    assert "product_ID" in df.columns
    val_report = validate_data(df)
    assert val_report["passed"] is True
    assert len(val_report["issues"]) == 0

    summary = get_dataset_summary(df)
    assert summary["n_missing"] == 0
    assert summary["target_distribution"][1] == 1400


def test_metrics_computation():
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 0, 1, 0, 1, 1, 1, 0])
    y_prob = np.array([0.1, 0.2, 0.9, 0.4, 0.85, 0.6, 0.75, 0.3])

    m = compute_all_metrics(y_true, y_pred, y_prob)
    assert 0.0 <= m["accuracy"] <= 1.0
    assert 0.0 <= m["fake_f1"] <= 1.0
    assert 0.0 <= m["macro_f1"] <= 1.0
    assert 0.0 <= m["roc_auc"] <= 1.0
    assert 0.0 <= m["pr_auc"] <= 1.0
    assert m["confusion_matrix"][1][1] == 3  # TP


def test_risk_score_and_adjusted_rating():
    # Low risk
    score_low, tier_low = compute_risk_score(0.15)
    assert score_low == 15.0
    assert tier_low == "LOW_RISK"

    # High risk
    score_high, tier_high = compute_risk_score(0.85)
    assert score_high == 85.0
    assert tier_high == "HIGH_RISK"

    # Adjusted rating
    adj_low = compute_trust_adjusted_rating(displayed_rating=4.8, p_fake=0.10, reference_rating=3.5)
    assert adj_low > 4.5  # Minimal discount

    adj_high = compute_trust_adjusted_rating(displayed_rating=4.8, p_fake=0.90, reference_rating=3.5)
    assert adj_high < 4.0  # Heavy discount down toward baseline


def test_synthetic_graph_construction():
    df_rev, ring_revs, target_prods = generate_synthetic_review_ring(
        n_ring_reviewers=5,
        n_target_products=3,
        n_honest_reviewers=20,
    )
    P = build_projected_product_graph(df_rev)
    assert P.number_of_nodes() > 0
    assert P.number_of_edges() > 0

    feats = extract_graph_features(P, include_advanced=False)
    assert "pagerank" in feats.columns
    assert "clustering_coef" in feats.columns
    assert "w_degree" in feats.columns
    assert len(feats) == P.number_of_nodes()
