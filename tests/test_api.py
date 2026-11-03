"""
True HTTP Integration and Unit Tests for FastAPI Application using TestClient.

Tests:
1. Health endpoint (GET /health)
2. Datasets summary (GET /datasets/summary)
3. Model metrics query (GET /models/metrics)
4. Valid prediction (POST /predict/product)
5. Incomplete feature rejection (HTTP 422 Unprocessable Entity)
6. Invalid numeric value rejection (HTTP 422)
7. Disallowed model rejection (HTTP 400 Bad Request)
8. Public product catalog lookup without ground-truth label leakage (GET /products/{id})
9. Nonexistent product lookup (HTTP 404 Not Found)
10. Clustering summary inspection (GET /clustering/summary)
"""

from __future__ import annotations

import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.api.main import app

client = TestClient(app)


def test_http_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == "2.1.0"
    assert "timestamp" in data


def test_http_datasets_summary():
    response = client.get("/datasets/summary")
    assert response.status_code == 200
    data = response.json()
    assert data["labeled_products"] == 3327
    assert data["positive_class_count"] == 1400
    assert data["fake_prevalence_pct"] > 40.0
    assert len(data["available_feature_groups"]) >= 3


def test_http_models_metrics():
    response = client.get("/models/metrics")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    sample = data[0]
    assert "accuracy" in sample
    assert "fake_f1" in sample
    assert "roc_auc" in sample


def test_http_predict_valid_payload():
    payload = {
        "product_ID": 12345,
        "displayed_rating": 4.8,
        "model_name": "review_network_xgboost",
        "features": {
            # 12 Review Features
            "tfidf_review_body": 0.45,
            "n_of_reviews": 60.0,
            "avg_review_rating": 4.8,
            "avg_days_between_reviews": 2.1,
            "stdev_days_between_reviews": 3.4,
            "max_days_between_reviews": 15.0,
            "min_days_between_reviews": 0.0,
            "share_helpful_reviews": 0.30,
            "share_1star": 0.05,
            "share_5star": 0.88,
            "share_photo": 0.15,
            "std_review_len": 45.2,
            # 4 Network Features
            "pagerank": 0.0003,
            "w_degree": 25.0,
            "clustering_coef": 0.45,
            "eigenvector_cent": 0.06,
        },
    }
    response = client.post("/predict/product", json=payload)
    assert response.status_code == 200
    res = response.json()
    assert res["product_ID"] == 12345
    assert "p_fake" in res
    assert 0.0 <= res["p_fake"] <= 1.0
    assert 0.0 <= res["risk_score"] <= 100.0
    assert res["risk_tier"] in ["LOW_RISK", "MODERATE_RISK", "HIGH_RISK"]
    assert 1.0 <= res["trust_adjusted_rating"] <= 5.0
    assert len(res["explanation_english"]) > 0
    assert len(res["explanation_hindi"]) > 0
    assert len(res["explanation_hinglish"]) > 0


def test_http_predict_28_features_multimodal():
    image_feats = {
        "min_sim": 0.3, "max_sim": 0.9, "mean_sim": 0.6, "std_sim": 0.15,
        "min_sim_review": 0.2, "max_sim_review": 0.8, "mean_sim_review": 0.5, "std_sim_review": 0.12,
        "min_sim_product": 0.4, "max_sim_product": 0.95, "mean_sim_product": 0.7, "std_sim_product": 0.1,
    }
    features = {
        "tfidf_review_body": 0.45, "n_of_reviews": 60.0, "avg_review_rating": 4.8,
        "avg_days_between_reviews": 2.1, "stdev_days_between_reviews": 3.4,
        "max_days_between_reviews": 15.0, "min_days_between_reviews": 0.0,
        "share_helpful_reviews": 0.30, "share_1star": 0.05, "share_5star": 0.88,
        "share_photo": 0.15, "std_review_len": 45.2,
        "pagerank": 0.0003, "w_degree": 25.0, "clustering_coef": 0.45, "eigenvector_cent": 0.06,
        **image_feats,
    }
    payload = {
        "product_ID": 54321,
        "displayed_rating": 4.5,
        "model_name": "review_network_image_xgboost",
        "features": features,
    }
    response = client.post("/predict/product", json=payload)
    assert response.status_code == 200
    res = response.json()
    assert res["model_name"] == "review_network_image_xgboost"
    assert 0.0 <= res["p_fake"] <= 1.0


def test_http_predict_image_model_missing_image_features_rejects_422():
    # Only 16 features provided to a 28-feature model -> MUST reject with 422 (no silent zero-filling)
    payload = {
        "displayed_rating": 4.5,
        "model_name": "review_network_image_xgboost",
        "features": {
            "tfidf_review_body": 0.45, "n_of_reviews": 60.0, "avg_review_rating": 4.8,
            "avg_days_between_reviews": 2.1, "stdev_days_between_reviews": 3.4,
            "max_days_between_reviews": 15.0, "min_days_between_reviews": 0.0,
            "share_helpful_reviews": 0.30, "share_1star": 0.05, "share_5star": 0.88,
            "share_photo": 0.15, "std_review_len": 45.2,
            "pagerank": 0.0003, "w_degree": 25.0, "clustering_coef": 0.45, "eigenvector_cent": 0.06,
        },
    }
    response = client.post("/predict/product", json=payload)
    assert response.status_code == 422
    assert "Missing" in response.json()["detail"]


def test_http_predict_review_only_12_features():
    payload = {
        "displayed_rating": 4.0,
        "model_name": "review_only_xgboost",
        "features": {
            "tfidf_review_body": 0.45, "n_of_reviews": 60.0, "avg_review_rating": 4.0,
            "avg_days_between_reviews": 2.1, "stdev_days_between_reviews": 3.4,
            "max_days_between_reviews": 15.0, "min_days_between_reviews": 0.0,
            "share_helpful_reviews": 0.30, "share_1star": 0.05, "share_5star": 0.88,
            "share_photo": 0.15, "std_review_len": 45.2,
        },
    }
    response = client.post("/predict/product", json=payload)
    assert response.status_code == 200
    assert response.json()["model_name"] == "review_only_xgboost"


def test_http_predict_network_only_4_features():
    payload = {
        "displayed_rating": 4.0,
        "model_name": "network_only_xgboost",
        "features": {
            "pagerank": 0.0003, "w_degree": 25.0, "clustering_coef": 0.45, "eigenvector_cent": 0.06,
        },
    }
    response = client.post("/predict/product", json=payload)
    assert response.status_code == 200
    assert response.json()["model_name"] == "network_only_xgboost"


def test_http_model_registry():
    response = client.get("/models/registry")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 15
    model_names = [m["model_name"] for m in data]
    assert "review_only_xgboost" in model_names
    assert "network_only_xgboost" in model_names
    assert "review_network_xgboost" in model_names
    assert "review_network_image_xgboost" in model_names
    # Verify required feature schema
    rev_model = next(m for m in data if m["model_name"] == "review_only_xgboost")
    assert rev_model["required_features_count"] == 12
    net_model = next(m for m in data if m["model_name"] == "network_only_xgboost")
    assert net_model["required_features_count"] == 4
    rn_model = next(m for m in data if m["model_name"] == "review_network_xgboost")
    assert rn_model["required_features_count"] == 16
    rni_model = next(m for m in data if m["model_name"] == "review_network_image_xgboost")
    assert rni_model["required_features_count"] == 28


def test_http_predict_missing_core_features_rejects_422():
    # Payload missing core review/network features
    incomplete_payload = {
        "displayed_rating": 4.5,
        "features": {
            "w_degree": 10.0,
            # Missing remaining 15 core features
        },
    }
    response = client.post("/predict/product", json=incomplete_payload)
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


def test_http_predict_invalid_rating_range_rejects_422():
    # Rating out of bounds (6.5 > 5.0)
    invalid_payload = {
        "displayed_rating": 6.5,
        "features": {"w_degree": 10.0},
    }
    response = client.post("/predict/product", json=invalid_payload)
    assert response.status_code == 422


def test_http_predict_disallowed_model_rejects_400():
    payload = {
        "displayed_rating": 4.5,
        "model_name": "malicious_or_unknown_model_path",
        "features": {
            "tfidf_review_body": 0.4, "n_of_reviews": 50.0, "avg_review_rating": 4.5,
            "avg_days_between_reviews": 2.0, "stdev_days_between_reviews": 2.0,
            "max_days_between_reviews": 10.0, "min_days_between_reviews": 0.0,
            "share_helpful_reviews": 0.2, "share_1star": 0.05, "share_5star": 0.8,
            "share_photo": 0.1, "std_review_len": 30.0, "pagerank": 0.001,
            "w_degree": 10.0, "clustering_coef": 0.2, "eigenvector_cent": 0.05,
        },
    }
    response = client.post("/predict/product", json=payload)
    assert response.status_code == 400
    assert "not permitted" in response.json()["detail"]


def test_http_product_details_protects_ground_truth():
    from src.data.data_loader import load_product_data
    df = load_product_data(with_images=True)
    first_pid = int(df["product_ID"].iloc[0])

    response = client.get(f"/products/{first_pid}")
    assert response.status_code == 200
    data = response.json()
    assert data["product_ID"] == first_pid
    # CRITICAL SECURITY CHECK: Ensure ground truth labels are NOT exposed to consumers
    assert "ground_truth_label" not in data
    assert "ground_truth_binary" not in data
    assert "fake" not in data
    assert "risk_score" in data
    assert "trust_adjusted_rating" in data


def test_http_product_not_found_returns_404():
    response = client.get("/products/999999999")
    assert response.status_code == 404


def test_http_clustering_summary():
    response = client.get("/clustering/summary")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    assert "available_k_values" in data
    assert "selected_k" in data
    assert "selection_criterion" in data
    assert "silhouette_score" in data
    assert "davies_bouldin_score" in data
    assert "calinski_harabasz_score" in data
    assert "cluster_count" in data
    assert "dataset_identifier" in data
    assert "result_source" in data
    assert data["selected_k"] == 2
    assert "cluster_profiles" in data
    assert len(data["cluster_profiles"]) > 0
    assert "cluster_ID" in data["cluster_profiles"][0]


def test_http_predict_calibrated_model():
    image_feats = {
        "min_sim": 0.3, "max_sim": 0.9, "mean_sim": 0.6, "std_sim": 0.15,
        "min_sim_review": 0.2, "max_sim_review": 0.8, "mean_sim_review": 0.5, "std_sim_review": 0.12,
        "min_sim_product": 0.4, "max_sim_product": 0.95, "mean_sim_product": 0.7, "std_sim_product": 0.1,
    }
    features = {
        "tfidf_review_body": 0.45, "n_of_reviews": 60.0, "avg_review_rating": 4.8,
        "avg_days_between_reviews": 2.1, "stdev_days_between_reviews": 3.4,
        "max_days_between_reviews": 15.0, "min_days_between_reviews": 0.0,
        "share_helpful_reviews": 0.30, "share_1star": 0.05, "share_5star": 0.88,
        "share_photo": 0.15, "std_review_len": 45.2,
        "pagerank": 0.0003, "w_degree": 25.0, "clustering_coef": 0.45, "eigenvector_cent": 0.06,
        **image_feats,
    }
    payload = {
        "product_ID": 9988,
        "displayed_rating": 4.5,
        "model_name": "calibrated_review_network_image_xgboost",
        "features": features,
    }
    response = client.post("/predict/product", json=payload)
    assert response.status_code == 200
    res = response.json()
    assert res["model_version"] == "2.1.0"
    assert res["probability_type"] == "calibrated"
    assert res["calibration_method"] in ["isotonic", "sigmoid"]
    assert 0.0 <= res["p_fake"] <= 1.0
    assert 0.0 <= res["risk_score"] <= 100.0


def test_http_predict_all_registered_models():
    """Verify that EVERY single registered model variant predicts successfully with valid schema."""
    from src.api.main import MODEL_REGISTRY
    from src.config import ALL_STRUCTURED_FEATURES

    all_feats = {f: 0.5 for f in ALL_STRUCTURED_FEATURES}
    all_feats["avg_review_rating"] = 4.2
    all_feats["n_of_reviews"] = 50.0

    assert len(MODEL_REGISTRY) >= 19, f"Expected at least 19 models, got {len(MODEL_REGISTRY)}"

    for model_name, info in MODEL_REGISTRY.items():
        req_feats = info["required_features"]
        model_payload = {
            "product_ID": 1234,
            "displayed_rating": 4.2,
            "model_name": model_name,
            "features": {f: all_feats[f] for f in req_feats},
        }
        res = client.post("/predict/product", json=model_payload)
        assert res.status_code == 200, f"Model {model_name} failed: {res.text}"
        data = res.json()
        assert "p_fake" in data
        assert 0.0 <= data["p_fake"] <= 1.0
        assert data["probability_type"] in ["calibrated", "raw"]
        assert data["calibration_method"] in ["isotonic", "sigmoid", "none"]
        assert 0.0 <= data["risk_score"] <= 100.0
        assert data["risk_tier"] in ["LOW_RISK", "MODERATE_RISK", "HIGH_RISK"]


def test_http_dashboard_overview():
    res = client.get("/dashboard/overview")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "dataset_statistics" in data
    assert data["dataset_statistics"]["total_products"] == 3327
    assert "model_statistics" in data
    assert data["model_statistics"]["registered_models_count"] == 21
    assert "feature_groups" in data
    assert "system_status" in data


def test_http_products_list():
    res = client.get("/products?page=1&page_size=10")
    assert res.status_code == 200
    data = res.json()
    assert data["total"] == 3327
    assert len(data["items"]) == 10
    # Must protect TARGET label
    assert "TARGET" not in data["items"][0]
    assert "fake" not in data["items"][0]


def test_http_experiments_registry():
    res = client.get("/experiments?page=1&page_size=20")
    assert res.status_code == 200
    data = res.json()
    assert data["total"] > 0
    assert len(data["records"]) <= 20
    assert "experiment_id" in data["records"][0]
    assert "algorithm" in data["records"][0]


def test_http_explainability_global():
    res = client.get("/explainability/global")
    assert res.status_code == 200
    data = res.json()
    assert "features" in data
    assert len(data["features"]) == 28
    assert "modality_breakdown" in data
    assert data["top_feature"] in ["clustering_coef", "eigenvector_cent"]


def test_http_clustering_details():
    res = client.get("/clustering/details")
    assert res.status_code == 200
    data = res.json()
    assert len(data["sweep_metrics"]) >= 10
    assert len(data["k2_profiles"]) == 2
    assert data["optimal_k"] == 2


def test_http_network_endpoints():
    res_sum = client.get("/network/summary")
    assert res_sum.status_code == 200
    data_sum = res_sum.json()
    assert len(data_sum["benchmarks"]) >= 5

    res_graph = client.get("/network/graph")
    assert res_graph.status_code == 200
    data_graph = res_graph.json()
    assert "nodes" in data_graph
    assert "edges" in data_graph
    assert len(data_graph["nodes"]) > 0
    assert len(data_graph["edges"]) > 0


def test_http_system_environment():
    res = client.get("/system/environment")
    assert res.status_code == 200
    data = res.json()
    assert "python_version" in data
    assert "sklearn_version" in data
    assert "xgboost_version" in data
    assert "catboost_version" in data
    assert "fastapi_version" in data
    assert "platform" in data
    assert data["docker_supported_python"] == "3.11"


def test_http_system_test_report():
    res = client.get("/system/test-report")
    assert res.status_code == 200
    data = res.json()
    assert "status" in data
    assert data["status"] in ["available", "passed", "failed", "not_available"]

