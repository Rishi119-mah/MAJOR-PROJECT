"""
Unit, Pipeline, and Integration Tests for Fake Review Detection.

DESIGN PRINCIPLES:
1. Deterministic Fixtures: Unit tests run without requiring large external dataset files.
2. Graceful Integration Marking: Tests requiring large real datasets use explicit pytest.mark.skipif
   with clear explanatory reasons, avoiding silent false passes or unexpected crashes.
3. Complete Pipeline Coverage: Data loading, target validation, train/test splitting,
   StandardScaler fitting, classification, calibration, risk scoring, and graph ML.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config import DATA_WITH_IMG, DATA_UCSD, REVIEW_NETWORK_FEATURES, TARGET
from src.data.data_loader import load_product_data, validate_data, get_dataset_summary
from src.evaluation.metrics import compute_all_metrics
from src.evaluation.risk_score import (
    compute_risk_score,
    compute_trust_adjusted_rating,
    compute_expected_calibration_error,
    evaluate_calibration_methods,
)
from src.graph.synthetic_rings import generate_synthetic_review_ring
from src.graph.graph_builder import (
    build_bipartite_graph,
    build_projected_product_graph,
    extract_graph_features,
)
from src.graph.gnn_models import (
    TrainableGraphSAGE,
    generate_node2vec_walks,
    train_node2vec_embeddings,
    KNNNeighborAggregator,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


@pytest.fixture
def synthetic_product_data() -> pd.DataFrame:
    """Deterministic synthetic fixture for unit tests without external data dependencies."""
    rng = np.random.default_rng(42)
    n = 60
    data = {
        "product_ID": np.arange(1001, 1001 + n),
        "fake": rng.choice([0, 1], size=n, p=[0.6, 0.4]),
        "tfidf_review_body": rng.uniform(0.1, 0.9, size=n),
        "n_of_reviews": rng.integers(5, 200, size=n).astype(float),
        "avg_review_rating": rng.uniform(2.5, 5.0, size=n),
        "avg_days_between_reviews": rng.uniform(0.5, 10.0, size=n),
        "stdev_days_between_reviews": rng.uniform(0.2, 5.0, size=n),
        "max_days_between_reviews": rng.uniform(5.0, 50.0, size=n),
        "min_days_between_reviews": rng.uniform(0.0, 2.0, size=n),
        "share_helpful_reviews": rng.uniform(0.0, 0.8, size=n),
        "share_1star": rng.uniform(0.0, 0.3, size=n),
        "share_5star": rng.uniform(0.3, 0.95, size=n),
        "share_photo": rng.uniform(0.0, 0.4, size=n),
        "std_review_len": rng.uniform(10.0, 100.0, size=n),
        "pagerank": rng.uniform(0.0001, 0.002, size=n),
        "w_degree": rng.uniform(1.0, 80.0, size=n),
        "clustering_coef": rng.uniform(0.0, 0.8, size=n),
        "eigenvector_cent": rng.uniform(0.0, 0.25, size=n),
    }
    return pd.DataFrame(data)


def test_deterministic_fixture_validation(synthetic_product_data):
    val_report = validate_data(synthetic_product_data)
    assert val_report["passed"] is True
    assert len(val_report["issues"]) == 0
    assert len(synthetic_product_data) == 60


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
    assert m["tp"] == 3
    assert m["tn"] == 3
    assert m["fp"] == 1
    assert m["fn"] == 1
    assert m["specificity"] == 0.75
    assert m["false_positive_rate"] == 0.25
    assert 0.0 <= m["expected_calibration_error"] <= 1.0


def test_risk_score_and_boundaries():
    # Test strict boundary logic
    score_low, tier_low = compute_risk_score(0.29)
    assert score_low == 29.0
    assert tier_low == "LOW_RISK"

    score_boundary_mod, tier_boundary_mod = compute_risk_score(0.30)
    assert score_boundary_mod == 30.0
    assert tier_boundary_mod == "MODERATE_RISK"

    score_mod, tier_mod = compute_risk_score(0.64)
    assert score_mod == 64.0
    assert tier_mod == "MODERATE_RISK"

    score_high, tier_high = compute_risk_score(0.65)
    assert score_high == 65.0
    assert tier_high == "HIGH_RISK"

    # Test trust-adjusted rating shrinkage
    adj_low = compute_trust_adjusted_rating(displayed_rating=4.9, p_fake=0.05, reference_rating=3.5)
    assert adj_low > 4.8  # Minimal discount for authentic product

    adj_high = compute_trust_adjusted_rating(displayed_rating=4.9, p_fake=0.95, reference_rating=3.5)
    assert adj_high < 3.6  # Heavy discount towards reference baseline


def test_probability_calibration_evaluation(synthetic_product_data, tmp_path):
    features = REVIEW_NETWORK_FEATURES
    X = synthetic_product_data[features].values
    y = synthetic_product_data[TARGET].values

    X_dev, X_test = X[:40], X[40:]
    y_dev, y_test = y[:40], y[40:]

    base_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)),
    ])

    custom_out = tmp_path / "test_cal.csv"
    cal_res, best_model = evaluate_calibration_methods(
        base_pipe, X_dev, y_dev, X_test, y_test, output_path=custom_out
    )
    assert "summary_df" in cal_res
    assert len(cal_res["summary_df"]) == 3  # Raw, Sigmoid, Isotonic
    assert hasattr(best_model, "predict_proba")
    assert custom_out.exists(), "Calibration evaluation must write to provided output_path"


def test_bipartite_graph_collision_prevention():
    # Create reviewer and product IDs with identical strings to test prefix collision safety
    overlap_reviews = pd.DataFrame({
        "reviewer_ID": ["ID_101", "ID_102", "ID_103"],
        "product_ID": ["ID_101", "ID_102", "ID_104"],
    })
    B, rev_set, prod_set = build_bipartite_graph(overlap_reviews, add_prefix=True)
    # Ensure no collision occurred: total nodes should be 3 rev + 3 prod = 6
    assert B.number_of_nodes() == 6
    assert "rev_ID_101" in B.nodes
    assert "prod_ID_101" in B.nodes
    assert B.has_edge("rev_ID_101", "prod_ID_101")


def test_trainable_graphsage_neural_network():
    rng = np.random.default_rng(42)
    n, d = 25, 8
    X = rng.normal(0, 1, (n, d))
    adj = (rng.uniform(0, 1, (n, n)) > 0.85).astype(float)
    np.fill_diagonal(adj, 0)
    adj = np.maximum(adj, adj.T)
    y = rng.choice([0, 1], size=n)

    train_mask = np.zeros(n, dtype=bool)
    train_mask[:18] = True
    val_mask = ~train_mask

    gnn = TrainableGraphSAGE(in_dim=d, hidden_dim=12, out_dim=6, lr=0.02, seed=42)
    history = gnn.fit(X, adj, y, train_mask=train_mask, val_mask=val_mask, max_epochs=15, patience=5)
    assert len(history["train_loss"]) > 0

    probs = gnn.predict_proba(X, adj)
    assert len(probs) == n
    assert np.all((probs >= 0.0) & (probs <= 1.0))

    embs = gnn.extract_embeddings(X, adj)
    assert embs.shape == (n, 6)


def test_node2vec_skipgram_embeddings():
    df_rev, _, _ = generate_synthetic_review_ring(n_ring_reviewers=4, n_target_products=3, n_honest_reviewers=10)
    P = build_projected_product_graph(df_rev)
    assert P.number_of_nodes() > 0

    walks = generate_node2vec_walks(P, num_walks=4, walk_length=8, seed=42)
    assert len(walks) > 0

    embs = train_node2vec_embeddings(walks, dim=8, window_size=2, epochs=3, seed=42)
    assert len(embs) == P.number_of_nodes()
    for pid, vec in embs.items():
        assert len(vec) == 8


@pytest.mark.skipif(not DATA_WITH_IMG.exists(), reason="Product dataset with images not present locally")
def test_real_dataset_loading_integration():
    df = load_product_data(with_images=True)
    assert len(df) == 3327
    assert TARGET in df.columns
    assert "product_ID" in df.columns
    summary = get_dataset_summary(df)
    assert summary["target_distribution"][1] == 1400


@pytest.mark.skipif(not DATA_UCSD.exists(), reason="UCSD product dataset not present locally")
def test_ucsd_dataset_loading_integration():
    from src.data.data_loader import load_ucsd_data
    df = load_ucsd_data()
    assert len(df) == 64585


def test_standalone_demo_workflow():
    from src.demo import run_demo
    result = run_demo()
    assert result["status"] == "success"
    assert len(result["responses"]) == 2
    for r in result["responses"]:
        assert "p_fake" in r
        assert 0.0 <= r["p_fake"] <= 1.0
        assert 0.0 <= r["risk_score"] <= 100.0
        assert r["risk_tier"] in ["LOW_RISK", "MODERATE_RISK", "HIGH_RISK"]
        assert 1.0 <= r["trust_adjusted_rating"] <= 5.0
        assert len(r["explanation_english"]) > 0


def test_holdout_isolation_regression(synthetic_product_data):
    """REGRESSION TEST: Verify that final test set cannot influence model or calibration selection."""
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import RandomForestClassifier

    features = REVIEW_NETWORK_FEATURES
    X = synthetic_product_data[features].values
    y = synthetic_product_data[TARGET].values

    # Dev (first 40) vs Test (last 20)
    X_dev, X_test = X[:40], X[40:]
    y_dev, y_test = y[:40], y[40:]

    clf = RandomForestClassifier(n_estimators=10, random_state=42)

    # 1. Selection score on clean dev set
    dev_score_1 = np.mean(cross_val_score(clf, X_dev, y_dev, cv=3, scoring="f1"))

    # 2. Perturb test set completely (flip labels and corrupt features)
    X_test_corrupt = X_test * 100.0
    y_test_corrupt = 1 - y_test

    # 3. Selection score on dev set must be 100% IDENTICAL (zero information leakage)
    dev_score_2 = np.mean(cross_val_score(clf, X_dev, y_dev, cv=3, scoring="f1"))
    assert dev_score_1 == dev_score_2, "Development CV model selection was influenced by external state!"

    # 4. Calibration selection must also be strictly invariant to test set corruption
    base_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)),
    ])
    res_clean, _ = evaluate_calibration_methods(base_pipe, X_dev, y_dev, X_test, y_test)
    res_corrupt, _ = evaluate_calibration_methods(base_pipe, X_dev, y_dev, X_test_corrupt, y_test_corrupt)

    assert res_clean["best_method"] == res_corrupt["best_method"], \
        "Calibration method selection changed when test set was corrupted! Must be selected on Dev only."


def test_calibration_isolation_and_persistence(synthetic_product_data, tmp_path):
    """Verify calibrated pipeline can be saved, reloaded, and unit test does not touch research files."""
    import joblib
    from src.config import EXPERIMENT_DIR
    real_csv = EXPERIMENT_DIR / "CALIBRATION_EVALUATION.csv"
    orig_mtime = real_csv.stat().st_mtime if real_csv.exists() else None

    features = REVIEW_NETWORK_FEATURES
    X = synthetic_product_data[features].values
    y = synthetic_product_data[TARGET].values
    X_dev, X_test = X[:40], X[40:]
    y_dev, y_test = y[:40], y[40:]

    base_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)),
    ])

    test_out = tmp_path / "custom_calibration.csv"
    cal_res, cal_model = evaluate_calibration_methods(
        base_pipe, X_dev, y_dev, X_test, y_test, output_path=test_out
    )

    # Persistence verification
    model_save_path = tmp_path / "test_calibrated_pipeline.joblib"
    joblib.dump(cal_model, model_save_path)
    reloaded_model = joblib.load(model_save_path)

    reloaded_probs = reloaded_model.predict_proba(X_test)[:, 1]
    assert len(reloaded_probs) == len(X_test)
    assert np.all((reloaded_probs >= 0.0) & (reloaded_probs <= 1.0))

    # Research output protection verification
    if real_csv.exists() and orig_mtime is not None:
        assert real_csv.stat().st_mtime == orig_mtime, \
            "Unit test accidentally modified production research file CALIBRATION_EVALUATION.csv!"
