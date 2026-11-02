# API_CONTRACT.md — FastAPI Service Specification

## 1. Overview
The Fake Review Detection API provides real-time inferences, graph neighborhood analysis, SHAP explainability, and access to all benchmark metrics.

Base URL: `http://localhost:8000/api/v1`

---

## 2. Endpoints

### 2.1 Health & Meta
- `GET /health`
  - Response: `{"status": "ok", "version": "1.0.0", "timestamp": "..."}`
- `GET /datasets/summary`
  - Returns product counts, label distribution, feature counts for all datasets.

### 2.2 Model Benchmarks
- `GET /models`
  - Returns list of available models and configurations.
- `GET /models/metrics`
  - Query params: `feature_group` (optional), `model_name` (optional)
  - Returns benchmark results table from `EXPERIMENT_RESULTS.csv`.

### 2.3 Product Predictions & Risk Scoring
- `POST /predict/product`
  - Request body:
    ```json
    {
      "features": {
        "tfidf_review_body": 0.05,
        "n_of_reviews": 45,
        "avg_review_rating": 4.6,
        "avg_days_between_reviews": 3.2,
        "stdev_days_between_reviews": 4.1,
        "max_days_between_reviews": 25.0,
        "min_days_between_reviews": 0.1,
        "share_helpful_reviews": 0.12,
        "share_1star": 0.05,
        "share_5star": 0.78,
        "share_photo": 0.15,
        "std_review_len": 45.2,
        "pagerank": 0.00012,
        "w_degree": 14,
        "clustering_coef": 0.32,
        "eigenvector_cent": 0.015
      },
      "model_name": "xgboost"
    }
    ```
  - Response body:
    ```json
    {
      "product_ID": null,
      "model_name": "xgboost",
      "p_fake": 0.842,
      "is_fake_prediction": true,
      "risk_score": 84.2,
      "risk_tier": "HIGH_RISK",
      "trust_adjusted_rating": 3.82,
      "explanation_summary": "High network co-review density and concentrated 5-star ratings elevate risk."
    }
    ```

- `GET /products/{product_id}`
  - Returns existing product record, ground-truth label (if in labelled set), feature values, and pre-computed risk score.

### 2.4 Explainability
- `POST /explain/product`
  - Request body: same as `/predict/product` or `{"product_id": 12345}`
  - Response: Top positive and negative contributing features, SHAP values, base value, and localized explanation strings (English, Hindi, Hinglish).

### 2.5 Clustering & Graph
- `GET /clustering/summary`
  - Returns silhouette scores across k=2..20, cluster sizes, and model-estimated fake percentages.
- `GET /graph/product/{product_id}/neighbors`
  - Returns adjacent product nodes in the co-review network, shared reviewer count (edge weights), and community ID.

---

## 3. Error Handling
- Standard HTTP status codes:
  - `200 OK`: Successful operation
  - `400 Bad Request`: Invalid inputs or missing feature keys
  - `404 Not Found`: Product ID not found in database
  - `500 Internal Server Error`: Pipeline processing failure

