# API_CONTRACT.md — FastAPI Service Specification (v2.1.0)

## 1. System Overview

The **Fake Review Detection REST API** provides production-grade, schema-validated inference, continuous risk scoring, Bayesian trust-adjusted rating shrinkage, multilingual explainability (English, Hindi, and Hinglish), and full experimental metrics for product-level fake-review-buyer classification.

- **Base URL**: `http://localhost:8000`
- **Interactive Documentation**: `http://localhost:8000/docs` (Swagger UI) / `http://localhost:8000/redoc`
- **Application Version**: `2.1.0`

### Security & Integrity Guarantees
1. **Model Allowlist**: Only pre-registered, validated model identifiers in `MODEL_REGISTRY` are permitted; arbitrary paths or unauthorized model names return HTTP 400 Bad Request.
2. **Model-Specific Feature Validation**: Payloads are strictly validated against the target model's expected feature schema (`review_only`: 12, `network_only`: 4, `review_network`: 16, `review_network_image`/`tuned`: 28). Incomplete payloads reject with HTTP 422 Unprocessable Content. Zero-filling of missing features is strictly prohibited.
3. **Protected Ground-Truth Labels**: Public product endpoints never expose training/evaluation ground-truth labels (`fake`, `ground_truth_label`) to client consumers.
4. **Reliable Probability Calibration**: Inference models output calibrated probabilities $[0.0, 1.0]$ evaluated via Expected Calibration Error (ECE) and Brier Score.
5. **CORS Whitelist**: Explicitly configured for frontend web dashboards (Next.js, React, Streamlit).

---

## 2. API Endpoints

### 2.1 Health Check
- **Endpoint**: `GET /health`
- **Description**: Verifies service liveness and returns version metadata.
- **Response** (`200 OK`):
  ```json
  {
    "status": "ok",
    "version": "2.1.0",
    "timestamp": "2026-11-02T21:40:00Z"
  }
  ```

---

### 2.2 Model Registry Catalog
- **Endpoint**: `GET /models/registry`
- **Description**: Returns all 21 approved production models, their required feature schemas, feature counts, and descriptions (including calibrated models).
- **Response** (`200 OK`):
  ```json
  [
    {
      "model_name": "calibrated_review_network_image_xgboost",
      "filename": "calibrated_review_network_image_xgboost_pipeline.joblib",
      "feature_group": "review_network_image",
      "required_features_count": 28,
      "required_features": [
        "tfidf_review_body", "n_of_reviews", "avg_review_rating",
        "avg_days_between_reviews", "stdev_days_between_reviews",
        "max_days_between_reviews", "min_days_between_reviews",
        "share_helpful_reviews", "share_1star", "share_5star",
        "share_photo", "std_review_len", "pagerank", "w_degree",
        "clustering_coef", "eigenvector_cent", "img_feat_0", "img_feat_1",
        "img_feat_2", "img_feat_3", "img_feat_4", "img_feat_5",
        "img_feat_6", "img_feat_7", "img_feat_8", "img_feat_9",
        "img_feat_10", "img_feat_11"
      ],
      "description": "Production calibrated XGBoost pipeline (Isotonic calibration on Development split, 28 multimodal features)."
    },
    {
      "model_name": "review_network_xgboost",
      "filename": "review_network_xgboost_pipeline.joblib",
      "feature_group": "review_network",
      "required_features_count": 16,
      "required_features": [
        "tfidf_review_body", "n_of_reviews", "avg_review_rating",
        "avg_days_between_reviews", "stdev_days_between_reviews",
        "max_days_between_reviews", "min_days_between_reviews",
        "share_helpful_reviews", "share_1star", "share_5star",
        "share_photo", "std_review_len", "pagerank", "w_degree",
        "clustering_coef", "eigenvector_cent"
      ],
      "description": "XGBoost classifier trained on 16 review and network structural features."
    }
  ]
  ```

---

### 2.3 Product Inference & Risk Scoring
- **Endpoint**: `POST /predict/product`
- **Description**: Executes product-level fake-buyer classification, computes calibrated probability, 0–100 risk score, trust-adjusted rating, and multilingual explainability.
- **Request Body**:
  ```json
  {
    "product_ID": 1080,
    "displayed_rating": 4.8,
    "model_name": "calibrated_review_network_xgboost",
    "features": {
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
      "pagerank": 0.0003,
      "w_degree": 25.0,
      "clustering_coef": 0.45,
      "eigenvector_cent": 0.06
    }
  }
  ```
- **Response** (`200 OK`):
  ```json
  {
    "product_ID": 1080,
    "model_name": "calibrated_review_network_xgboost",
    "model_version": "2.1.0",
    "probability_type": "calibrated",
    "calibration_method": "isotonic",
    "p_fake": 0.8954,
    "is_fake_prediction": true,
    "risk_score": 89.54,
    "risk_tier": "HIGH_RISK",
    "trust_adjusted_rating": 3.52,
    "explanation_english": "Primary factors elevating risk: Co-review network degree (+0.312); Reviewer network clustering coefficient (+0.245).",
    "hindi_explanation": "जोखिम बढ़ाने वाले मुख्य कारक: नेटवर्क को-रिव्यू डिग्री (+0.312); रिव्यूअर नेटवर्क क्लस्टरिंग गुणांक (+0.245)।",
    "hinglish_explanation": "Risk badhane wale main factors: Co-review network degree (+0.312); Reviewer network clustering coefficient (+0.245). Inka unusual pattern coordinated review buying indicate karta hai."
  }
  ```

#### Strict Risk Scoring Tiers:
- `[0.0, 30.0)`: `"LOW_RISK"`
- `[30.0, 65.0)`: `"MODERATE_RISK"`
- `[65.0, 100.0]`: `"HIGH_RISK"`

---

### 2.4 Catalog Product Profiling
- **Endpoint**: `GET /products/{product_id}`
- **Description**: Fetches public risk metrics and behavioral feature values for an indexed catalog product without exposing ground-truth evaluation labels.
- **Response** (`200 OK`):
  ```json
  {
    "product_ID": 1001,
    "displayed_rating": 4.25,
    "risk_score": 14.2,
    "risk_tier": "LOW_RISK",
    "trust_adjusted_rating": 4.18,
    "features": {
      "avg_review_rating": 4.25,
      "n_of_reviews": 34.0,
      "w_degree": 3.0,
      "clustering_coef": 0.08
    }
  }
  ```
- **Error Response** (`404 Not Found`): If product ID does not exist in catalog.

---

### 2.5 Dataset Metadata
- **Endpoint**: `GET /datasets/summary`
- **Description**: Returns summary statistics of the labelled evaluation datasets.
- **Response** (`200 OK`):
  ```json
  {
    "labeled_products": 3327,
    "features_count": 28,
    "positive_class_count": 1400,
    "negative_class_count": 1927,
    "fake_prevalence_pct": 42.08,
    "available_feature_groups": [
      "review_features (12)",
      "network_features (4)",
      "image_features (12)",
      "text_tfidf (1000)"
    ]
  }
  ```

---

### 2.6 Model Performance Benchmarks
- **Endpoint**: `GET /models/metrics`
- **Query Parameters**:
  - `model_name` *(optional string)*: Filter by model identifier
  - `feature_group` *(optional string)*: Filter by feature modality
- **Response** (`200 OK`):
  ```json
  [
    {
      "experiment_id": "improved_review_network_image",
      "model": "xgboost",
      "feature_group": "review_network_image",
      "accuracy": 0.8844,
      "fake_f1": 0.8642,
      "roc_auc": 0.9412,
      "pr_auc": 0.9328,
      "ece": 0.0412,
      "brier_score": 0.0915
    }
  ]
  ```

---

### 2.7 Clustering & Topology Summary
- **Endpoint**: `GET /clustering/summary`
- **Description**: Returns diagnostic cluster metrics, silhouette scores across $k=2 \dots 20$, and optimal configuration calculated dynamically without hardcoded defaults.
- **Response** (`200 OK`):
  ```json
  {
    "available_k_values": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    "selected_k": 2,
    "selection_criterion": "max_silhouette",
    "silhouette_score": 0.2474,
    "davies_bouldin_score": 1.5681,
    "calinski_harabasz_score": 590.8807,
    "cluster_count": 2,
    "dataset_identifier": "amazon_electronics_multimodal",
    "result_source": "KMEANS_SWEEP_METRICS.csv",
    "optimal_k": 2,
    "optimal_silhouette": 0.2474,
    "cluster_profiles": [
      {
        "cluster_ID": 0,
        "n_products": 196,
        "share_of_total": 0.0575,
        "ground_truth_fake_prevalence": 0.9082,
        "mean_p_fake_predicted": 0.9302,
        "mean_w_degree": 837.28,
        "mean_clustering_coef": 0.3694
      }
    ]
  }
  ```

---

### 2.8 System Runtime Environment
- **Endpoint**: `GET /system/environment`
- **Description**: Returns runtime Python and library dependencies version information.
- **Response** (`200 OK`):
  ```json
  {
    "python_version": "3.14.6",
    "full_python_version": "3.14.6 (v3.14.6:2e4f0dc0709) [Clang 15.0.0 (clang-1500.3.9.4)]",
    "sklearn_version": "1.9.1",
    "xgboost_version": "3.4.1",
    "catboost_version": "1.2.10",
    "fastapi_version": "0.162.0",
    "platform": "macOS-15.3-arm64-arm-64bit",
    "docker_supported_python": "3.11",
    "host_runtime_status": "verified"
  }
  ```

---

### 2.9 Automated Test Execution Report
- **Endpoint**: `GET /system/test-report`
- **Description**: Returns verified pytest run report from `outputs/test_report.json`.
- **Response** (`200 OK`):
  ```json
  {
    "status": "passed",
    "total_tests": 37,
    "passed": 37,
    "failed": 0,
    "skipped": 0,
    "duration_seconds": 3.49,
    "test_execution_date": "2026-11-03T05:03:59.695517+00:00",
    "environment": "Python 3.14.6 • pytest 9.2.0 • Darwin",
    "test_command": "pytest tests/ -v",
    "tests": [...]
  }
  ```

---

## 3. Error Handling and Status Codes

| Status Code | Description | Payload Example |
|-------------|-------------|-----------------|
| `200 OK` | Successful execution | Desired JSON schema |
| `400 BAD REQUEST` | Disallowed model identifier | `{"detail": "Model 'xyz' is not permitted. Allowed models: [...]"}` |
| `404 NOT FOUND` | Resource or model artifact not found | `{"detail": "Product with ID 999999 not found in catalog."}` |
| `422 UNPROCESSABLE CONTENT` | Missing required features or out-of-range rating | `{"detail": "Incomplete feature set for model 'review_network_xgboost'. Missing 1 required features: ['w_degree']"}` |
| `500 INTERNAL SERVER ERROR` | Model inference or processing error | `{"detail": "Inference error: ..."}` |
