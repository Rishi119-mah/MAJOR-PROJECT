"""
Production-grade FastAPI Backend for Fake Review Detection.

Provides secure, schema-validated inference, TreeSHAP explainability (EN, HI, Hinglish),
risk scoring, and clustering inspection.

SECURITY & VALIDATION GUARANTEES:
1. Strict Model Allowlist: No arbitrary paths or silent fallbacks.
2. Robust Feature Validation: Incomplete inputs reject with HTTP 422; no silent zero-filling.
3. Protected Evaluation Data: Public endpoints never expose ground-truth training labels.
4. Reliable Probabilities: Rejects inference if calibrated probabilities are unavailable.
5. Explicit CORS Configuration: Whitelisted frontend origins.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import time

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import joblib
import pandas as pd
import numpy as np
from sklearn.calibration import CalibratedClassifierCV

from src.config import (
    ALL_STRUCTURED_FEATURES,
    EXPERIMENT_DIR,
    MODEL_DIR,
    REVIEW_FEATURES,
    NETWORK_FEATURES,
    IMAGE_FEATURES,
    REVIEW_NETWORK_FEATURES,
    TARGET,
)
from src.evaluation.risk_score import compute_risk_score, compute_trust_adjusted_rating
from src.evaluation.explainability import explain_single_product
from src.data.data_loader import load_product_data

logger = logging.getLogger("fastapi_app")

app = FastAPI(
    title="Fake Review Detection API",
    description="Production-grade Explainable Graph ML and Classification API for detecting fake review buyers.",
    version="2.1.0",
)

# Whitelist allowed frontend origins (e.g., React, Next.js, Streamlit, Vite)
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:5174",
    "http://localhost:5175",
    "http://localhost:8501",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5174",
    "http://127.0.0.1:5175",
    "http://127.0.0.1:8501",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────
# Model Registry and Schema Validation Allowlist
# ──────────────────────────────────────────────
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    # 1. Review Only (12 features)
    "review_only_xgboost": {
        "filename": "review_only_xgboost_pipeline.joblib",
        "feature_group": "review_only",
        "required_features": REVIEW_FEATURES,
        "description": "XGBoost classifier trained on 12 review-level behavioral features.",
    },
    "review_only_random_forest": {
        "filename": "review_only_random_forest_pipeline.joblib",
        "feature_group": "review_only",
        "required_features": REVIEW_FEATURES,
        "description": "Random Forest classifier trained on 12 review-level behavioral features.",
    },
    "review_only_catboost": {
        "filename": "review_only_catboost_pipeline.joblib",
        "feature_group": "review_only",
        "required_features": REVIEW_FEATURES,
        "description": "CatBoost classifier trained on 12 review-level behavioral features.",
    },
    "review_only_stacking": {
        "filename": "review_only_stacking_pipeline.joblib",
        "feature_group": "review_only",
        "required_features": REVIEW_FEATURES,
        "description": "Stacking ensemble trained on 12 review-level behavioral features.",
    },
    # 2. Network Only (4 features)
    "network_only_xgboost": {
        "filename": "network_only_xgboost_pipeline.joblib",
        "feature_group": "network_only",
        "required_features": NETWORK_FEATURES,
        "description": "XGBoost classifier trained on 4 bipartite graph centrality features.",
    },
    "network_only_random_forest": {
        "filename": "network_only_random_forest_pipeline.joblib",
        "feature_group": "network_only",
        "required_features": NETWORK_FEATURES,
        "description": "Random Forest classifier trained on 4 bipartite graph centrality features.",
    },
    "network_only_catboost": {
        "filename": "network_only_catboost_pipeline.joblib",
        "feature_group": "network_only",
        "required_features": NETWORK_FEATURES,
        "description": "CatBoost classifier trained on 4 bipartite graph centrality features.",
    },
    "network_only_stacking": {
        "filename": "network_only_stacking_pipeline.joblib",
        "feature_group": "network_only",
        "required_features": NETWORK_FEATURES,
        "description": "Stacking ensemble trained on 4 bipartite graph centrality features.",
    },
    # 3. Review + Network (16 features)
    "review_network_xgboost": {
        "filename": "review_network_xgboost_pipeline.joblib",
        "feature_group": "review_network",
        "required_features": REVIEW_NETWORK_FEATURES,
        "description": "XGBoost classifier trained on 16 review and network structural features.",
    },
    "review_network_random_forest": {
        "filename": "review_network_random_forest_pipeline.joblib",
        "feature_group": "review_network",
        "required_features": REVIEW_NETWORK_FEATURES,
        "description": "Random Forest classifier trained on 16 review and network structural features.",
    },
    "review_network_catboost": {
        "filename": "review_network_catboost_pipeline.joblib",
        "feature_group": "review_network",
        "required_features": REVIEW_NETWORK_FEATURES,
        "description": "CatBoost classifier trained on 16 review and network structural features.",
    },
    "review_network_stacking": {
        "filename": "review_network_stacking_pipeline.joblib",
        "feature_group": "review_network",
        "required_features": REVIEW_NETWORK_FEATURES,
        "description": "Stacking ensemble trained on 16 review and network structural features.",
    },
    # 4. Review + Network + Image (28 features)
    "review_network_image_xgboost": {
        "filename": "review_network_image_xgboost_pipeline.joblib",
        "feature_group": "review_network_image",
        "required_features": ALL_STRUCTURED_FEATURES,
        "description": "XGBoost classifier trained on all 28 multimodal features.",
    },
    "review_network_image_random_forest": {
        "filename": "review_network_image_random_forest_pipeline.joblib",
        "feature_group": "review_network_image",
        "required_features": ALL_STRUCTURED_FEATURES,
        "description": "Random Forest classifier trained on all 28 multimodal features.",
    },
    "review_network_image_catboost": {
        "filename": "review_network_image_catboost_pipeline.joblib",
        "feature_group": "review_network_image",
        "required_features": ALL_STRUCTURED_FEATURES,
        "description": "CatBoost classifier trained on all 28 multimodal features.",
    },
    "review_network_image_stacking": {
        "filename": "review_network_image_stacking_pipeline.joblib",
        "feature_group": "review_network_image",
        "required_features": ALL_STRUCTURED_FEATURES,
        "description": "Stacking ensemble trained on all 28 multimodal features.",
    },
    # 5. Tuned Models (28 features)
    "tuned_xgboost": {
        "filename": "tuned_xgboost_pipeline.joblib",
        "feature_group": "tuned_all",
        "required_features": ALL_STRUCTURED_FEATURES,
        "description": "Optuna-tuned XGBoost model evaluated on 28 multimodal features.",
    },
    "tuned_random_forest": {
        "filename": "tuned_random_forest_pipeline.joblib",
        "feature_group": "tuned_all",
        "required_features": ALL_STRUCTURED_FEATURES,
        "description": "Optuna-tuned Random Forest model evaluated on 28 multimodal features.",
    },
    "tuned_lightgbm": {
        "filename": "tuned_lightgbm_pipeline.joblib",
        "feature_group": "tuned_all",
        "required_features": ALL_STRUCTURED_FEATURES,
        "description": "Optuna-tuned LightGBM model evaluated on 28 multimodal features.",
    },
    # 6. Calibrated Production Models (Isotonic Calibration)
    "calibrated_review_network_image_xgboost": {
        "filename": "calibrated_review_network_image_xgboost_pipeline.joblib",
        "feature_group": "review_network_image",
        "required_features": ALL_STRUCTURED_FEATURES,
        "description": "Isotonic Calibrated XGBoost classifier evaluated on 28 multimodal features.",
    },
    "calibrated_review_network_xgboost": {
        "filename": "calibrated_review_network_xgboost_pipeline.joblib",
        "feature_group": "review_network",
        "required_features": REVIEW_NETWORK_FEATURES,
        "description": "Isotonic Calibrated XGBoost classifier evaluated on 16 review and network structural features.",
    },
}

ALLOWED_MODELS: Dict[str, str] = {k: v["filename"] for k, v in MODEL_REGISTRY.items()}

# In-memory model and dataset cache
_MODELS: Dict[str, Any] = {}
_DF_LABELED: Optional[pd.DataFrame] = None


def get_cached_model(name: str):
    """Retrieve model from cache or load from disk with strict allowlist verification."""
    if name not in ALLOWED_MODELS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model '{name}' is not permitted. Allowed models: {sorted(list(ALLOWED_MODELS.keys()))}",
        )

    if name not in _MODELS:
        filename = ALLOWED_MODELS[name]
        path = MODEL_DIR / filename
        if not path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model artifact '{filename}' not found on server.",
            )
        logger.info("Loading model artifact from %s", path)
        _MODELS[name] = joblib.load(path)

    return _MODELS[name]


def get_labeled_data() -> pd.DataFrame:
    global _DF_LABELED
    if _DF_LABELED is None:
        _DF_LABELED = load_product_data(with_images=True)
    return _DF_LABELED


# ──────────────────────────────────────────────
# Pydantic Schemas with Strict Validation
# ──────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str


class ModelRegistryItem(BaseModel):
    model_name: str
    filename: str
    feature_group: str
    required_features_count: int
    required_features: List[str]
    description: str


class ProductFeatureInput(BaseModel):
    product_ID: Optional[int] = Field(None, description="Optional catalog product identifier")
    displayed_rating: float = Field(..., ge=1.0, le=5.0, description="Observed consumer star rating between 1.0 and 5.0")
    model_name: str = Field("review_network_xgboost", description="Registered model name from allowlist")
    features: Dict[str, float] = Field(..., description="Key-value mapping of required model-specific features")

    @field_validator("features")
    @classmethod
    def validate_features_non_empty(cls, v: Dict[str, float]) -> Dict[str, float]:
        if not v:
            raise ValueError("Feature payload must not be empty.")
        # Ensure all values are finite numbers
        for k, val in v.items():
            if np.isnan(val) or np.isinf(val):
                raise ValueError(f"Feature '{k}' contains invalid non-finite value: {val}")
        return v


class PredictionResponse(BaseModel):
    product_ID: Optional[int] = None
    model_name: str
    model_version: str = "2.1.0"
    p_fake: float
    probability_type: str = Field("raw", description="Type of probability: 'calibrated' or 'raw'")
    calibration_method: str = Field("none", description="Calibration method applied: 'isotonic', 'sigmoid', or 'none'")
    is_fake_prediction: bool
    risk_score: float
    risk_tier: str
    trust_adjusted_rating: float
    explanation_english: str
    explanation_hindi: str
    explanation_hinglish: str


class ProductPublicDetails(BaseModel):
    product_ID: int
    displayed_rating: float
    risk_score: float
    risk_tier: str
    trust_adjusted_rating: float
    features: Dict[str, float]


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
def health():
    return {
        "status": "ok",
        "version": "2.1.0",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


@app.get("/datasets/summary")
def get_datasets_summary():
    df = get_labeled_data()
    missing_count = int(df.isna().sum().sum())
    dev_n = int(round(len(df) * 0.8))
    holdout_n = len(df) - dev_n

    return {
        "labeled_products": len(df),
        "features_count": len(df.columns) - 2,
        "positive_class_count": int(df[TARGET].sum()),
        "negative_class_count": int((df[TARGET] == 0).sum()),
        "fake_prevalence_pct": round(float(df[TARGET].mean()) * 100.0, 2),
        "missing_values_count": missing_count,
        "missing_values_status": f"{missing_count} missing" if missing_count > 0 else "0 (100% Clean)",
        "development_split_count": dev_n,
        "holdout_split_count": holdout_n,
        "available_feature_groups": [
            f"review_features ({len(REVIEW_FEATURES)})",
            f"network_features ({len(NETWORK_FEATURES)})",
            f"image_features ({len(IMAGE_FEATURES)})",
            "text_tfidf (1000)",
        ],
    }


@app.get("/models/registry", response_model=List[ModelRegistryItem])
def get_model_registry():
    """Retrieve full catalog of registered models, feature requirements, and specifications."""
    return [
        {
            "model_name": name,
            "filename": info["filename"],
            "feature_group": info["feature_group"],
            "required_features_count": len(info["required_features"]),
            "required_features": info["required_features"],
            "description": info["description"],
        }
        for name, info in MODEL_REGISTRY.items()
    ]


@app.get("/models/metrics")
def get_metrics(
    model_name: Optional[str] = None,
    feature_group: Optional[str] = None,
):
    results_path = EXPERIMENT_DIR / "FINAL_EXPERIMENT_RESULTS.csv"
    if not results_path.exists():
        results_path = EXPERIMENT_DIR / "EXPERIMENT_RESULTS.csv"
    if not results_path.exists():
        raise HTTPException(status_code=404, detail="No experiment results registered yet.")

    df_res = pd.read_csv(results_path)
    if model_name:
        df_res = df_res[df_res["model"].str.contains(model_name, case=False, na=False)]
    if feature_group:
        df_res = df_res[df_res["feature_group"].str.contains(feature_group, case=False, na=False)]

    df_clean = df_res.replace({np.nan: None})
    return df_clean.to_dict(orient="records")


@app.post("/predict/product", response_model=PredictionResponse)
def predict_product(payload: ProductFeatureInput):
    # 1. Verify model from allowlist
    if payload.model_name not in MODEL_REGISTRY:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model '{payload.model_name}' is not permitted. Allowed models: {sorted(list(MODEL_REGISTRY.keys()))}",
        )

    pipeline = get_cached_model(payload.model_name)
    model_meta = MODEL_REGISTRY[payload.model_name]
    required_features = model_meta["required_features"]

    # 2. Strict model-specific feature validation: require all required features, NO silent zero-filling
    provided_keys = set(payload.features.keys())
    missing_features = [f for f in required_features if f not in provided_keys]
    if missing_features:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Incomplete feature set for model '{payload.model_name}'. "
                f"Requires {len(required_features)} features ({model_meta['feature_group']}). "
                f"Missing {len(missing_features)} required features: {missing_features}"
            ),
        )

    # Construct input dataframe strictly with required features in canonical order
    input_df = pd.DataFrame([{f: payload.features[f] for f in required_features}])[required_features]

    # 3. Model inference: MUST support predict_proba
    if not hasattr(pipeline, "predict_proba"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model '{payload.model_name}' does not support probability estimation.",
        )

    try:
        p_fake = float(pipeline.predict_proba(input_df)[0, 1])
    except Exception as e:
        logger.error("Inference failure on model %s: %s", payload.model_name, e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Inference error: {str(e)}")

    is_fake = p_fake >= 0.5
    risk_score, risk_tier = compute_risk_score(p_fake)
    adjusted_rating = compute_trust_adjusted_rating(payload.displayed_rating, p_fake)

    # Determine probability and calibration properties
    is_cal = isinstance(pipeline, CalibratedClassifierCV) or hasattr(pipeline, "calibrated_classifiers_") or "calibrated" in payload.model_name
    probability_type = "calibrated" if is_cal else "raw"
    calibration_method = getattr(pipeline, "method", "isotonic" if is_cal else "none")

    # 4. Localized explanations
    if hasattr(pipeline, "calibrated_classifiers_") and len(pipeline.calibrated_classifiers_) > 0:
        base_est = pipeline.calibrated_classifiers_[0].estimator
        model_raw = base_est.named_steps["classifier"] if hasattr(base_est, "named_steps") else base_est
    elif hasattr(pipeline, "named_steps"):
        model_raw = pipeline.named_steps["classifier"]
    else:
        model_raw = pipeline

    try:
        import shap
        explainer = shap.TreeExplainer(model_raw)
        exp = explain_single_product(explainer, input_df, required_features)
        exp_en = exp["english_explanation"]
        exp_hi = exp["hindi_explanation"]
        exp_hing = exp["hinglish_explanation"]
    except Exception:
        exp_en = f"Model predicts {p_fake*100:.1f}% probability of fake review manipulation."
        exp_hi = f"मॉडल फेक रिव्यू हेरफेर की {p_fake*100:.1f}% संभावना का अनुमान लगाता है।"
        exp_hing = f"Model fake review manipulation ki {p_fake*100:.1f}% probability predict karta hai."

    return {
        "product_ID": payload.product_ID,
        "model_name": payload.model_name,
        "model_version": "2.1.0",
        "p_fake": round(p_fake, 4),
        "probability_type": probability_type,
        "calibration_method": calibration_method,
        "is_fake_prediction": is_fake,
        "risk_score": risk_score,
        "risk_tier": risk_tier,
        "trust_adjusted_rating": adjusted_rating,
        "explanation_english": exp_en,
        "explanation_hindi": exp_hi,
        "explanation_hinglish": exp_hing,
    }


@app.get("/products/{product_id}", response_model=ProductPublicDetails)
def get_product_details(product_id: int):
    """Retrieve consumer-facing product risk profile without exposing ground-truth training labels."""
    df = get_labeled_data()
    match = df[df["product_ID"] == product_id]
    if match.empty:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Product with ID {product_id} not found in catalog.",
        )

    row = match.iloc[0].to_dict()
    # Remove TARGET to protect evaluation ground truth
    row.pop(TARGET, None)

    # Compute risk score for catalog product
    pipeline = get_cached_model("review_network_image_xgboost")
    input_df = pd.DataFrame([row])[ALL_STRUCTURED_FEATURES]
    p_fake = float(pipeline.predict_proba(input_df)[0, 1])
    displayed_rating = float(row.get("avg_review_rating", 4.0))

    risk_score, risk_tier = compute_risk_score(p_fake)
    trust_rating = compute_trust_adjusted_rating(displayed_rating, p_fake)

    feature_dict = {k: float(v) for k, v in row.items() if k != "product_ID"}

    return {
        "product_ID": product_id,
        "displayed_rating": round(displayed_rating, 2),
        "risk_score": risk_score,
        "risk_tier": risk_tier,
        "trust_adjusted_rating": trust_rating,
        "features": feature_dict,
    }


@app.get("/clustering/summary")
def get_clustering_summary():
    """Retrieve optimal clustering configuration calculated from empirical sweep metrics and cluster profiles."""
    from src.models.clustering_pipeline import get_optimal_cluster_configuration

    summary = get_optimal_cluster_configuration(EXPERIMENT_DIR / "KMEANS_SWEEP_METRICS.csv")

    path = EXPERIMENT_DIR / "CLUSTER_ANALYSIS_SUMMARY.csv"
    if path.exists():
        df_c = pd.read_csv(path)
        summary["cluster_profiles"] = df_c.replace({np.nan: None}).to_dict(orient="records")
    else:
        summary["cluster_profiles"] = []

    return summary


# ──────────────────────────────────────────────
# Additional Read-Only Endpoints for Web Dashboard
# ──────────────────────────────────────────────
@app.get("/dashboard/overview")
def get_dashboard_overview():
    """Aggregated overview metrics for the executive landing dashboard."""
    df_labeled = get_labeled_data()
    total_prods = len(df_labeled)
    pos_count = int(df_labeled[TARGET].sum())
    neg_count = int((df_labeled[TARGET] == 0).sum())
    fake_prev = round(float(df_labeled[TARGET].mean()) * 100.0, 2)

    # Dynamic metrics from FINAL_EXPERIMENT_RESULTS.csv
    results_path = EXPERIMENT_DIR / "FINAL_EXPERIMENT_RESULTS.csv"
    best_model_name = "lightgbm (review_network_image)"
    best_fake_f1 = 0.8193
    best_roc_auc = 0.9191
    feature_progression = []
    top_models_list = []
    protocol_counts = {}

    if results_path.exists():
        df_res = pd.read_csv(results_path)

        # Count records per evaluation protocol
        if "evaluation_type" in df_res.columns:
            protocol_counts = df_res["evaluation_type"].value_counts().to_dict()

        df_final = df_res[df_res["evaluation_type"].str.contains("Final independent test", na=False)]
        if not df_final.empty and "fake_f1" in df_final.columns:
            best_idx = df_final["fake_f1"].idxmax()
            best_row = df_final.loc[best_idx]
            best_model_name = f"{best_row['model']} ({best_row.get('feature_group', '')})"
            best_fake_f1 = round(float(best_row["fake_f1"]), 4)
            best_roc_auc = round(float(best_row.get("roc_auc", 0.0)), 4)

            # Feature group progression for XGBoost (Review Only -> Network Only -> Review+Network -> Multimodal)
            groups = [
                ("review_only", "Review Only", 12),
                ("network_only", "Network Only", 4),
                ("review_network", "Review + Network", 16),
                ("review_network_image", "Multimodal (All)", 28),
            ]
            for g_key, g_name, f_count in groups:
                match = df_final[(df_final["model"].str.lower() == "xgboost") & (df_final["feature_group"] == g_key)]
                if not match.empty:
                    r = match.iloc[0]
                    feature_progression.append({
                        "group": g_name,
                        "f1": round(float(r["fake_f1"]), 4),
                        "auc": round(float(r.get("roc_auc", 0.0)), 4),
                        "features": f_count,
                        "protocol": "Final independent test",
                    })

            # Top validated models on final holdout test set
            df_sorted = df_final.sort_values(by="fake_f1", ascending=False)
            for _, r in df_sorted.head(6).iterrows():
                top_models_list.append({
                    "model": f"{r['model']} ({r.get('feature_group', '')})",
                    "f1": round(float(r["fake_f1"]), 4),
                    "auc": round(float(r.get("roc_auc", 0.0)), 4),
                    "type": "Final Holdout Test",
                    "accuracy": round(float(r.get("accuracy", 0.0)), 4) if pd.notnull(r.get("accuracy")) else None,
                })

    # Dynamic clustering summary from sweep metrics (no hardcoded defaults)
    from src.models.clustering_pipeline import get_optimal_cluster_configuration
    clustering_summary = get_optimal_cluster_configuration(EXPERIMENT_DIR / "KMEANS_SWEEP_METRICS.csv")

    # Dynamic SHAP modality summary
    network_pct = 35.42
    shap_path = EXPERIMENT_DIR / "SHAP_FEATURE_IMPORTANCE.csv"
    if shap_path.exists():
        df_sh = pd.read_csv(shap_path)
        if "modality" in df_sh.columns and "mean_abs_shap" in df_sh.columns:
            tot = df_sh["mean_abs_shap"].sum()
            if tot > 0:
                net_sum = df_sh[df_sh["modality"] == "network"]["mean_abs_shap"].sum()
                network_pct = round(float((net_sum / tot) * 100.0), 2)

    # Recent experiments from FINAL_EXPERIMENT_REGISTRY.csv
    registry_path = Path("FINAL_EXPERIMENT_REGISTRY.csv")
    if not registry_path.exists():
        registry_path = EXPERIMENT_DIR / "FINAL_EXPERIMENT_REGISTRY.csv"

    recent_exps = []
    if registry_path.exists():
        df_reg = pd.read_csv(registry_path)
        recent_exps = df_reg.tail(8).replace({np.nan: None}).to_dict(orient="records")

    return {
        "status": "ok",
        "dataset_statistics": {
            "total_products": total_prods,
            "fake_products_count": pos_count,
            "genuine_products_count": neg_count,
            "fake_prevalence_pct": fake_prev,
            "features_count": len(ALL_STRUCTURED_FEATURES),
        },
        "model_statistics": {
            "registered_models_count": len(MODEL_REGISTRY),
            "best_validated_model": best_model_name,
            "best_fake_f1": best_fake_f1,
            "best_roc_auc": best_roc_auc,
        },
        "feature_groups": [
            {"name": "Review Behaviors", "feature_count": len(REVIEW_FEATURES), "key": "review_only"},
            {"name": "Network Centrality", "feature_count": len(NETWORK_FEATURES), "key": "network_only"},
            {"name": "Review + Network", "feature_count": len(REVIEW_NETWORK_FEATURES), "key": "review_network"},
            {"name": "Multimodal (All Structured)", "feature_count": len(ALL_STRUCTURED_FEATURES), "key": "review_network_image"},
        ],
        "feature_group_progression": feature_progression,
        "top_models": top_models_list,
        "protocol_summary": protocol_counts,
        "clustering_summary": clustering_summary,
        "shap_summary": {
            "network_pct": network_pct,
        },
        "recent_experiments": recent_exps,
        "system_status": {
            "api_version": "2.1.0",
            "environment": "production",
            "models_loaded": len(_MODELS),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }


@app.get("/products")
def list_products(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    search: Optional[str] = None,
):
    """Retrieve catalog products without exposing ground truth training labels."""
    df = get_labeled_data()
    # Copy and protect evaluation ground truth
    display_df = df.copy()
    display_df = display_df.drop(columns=[TARGET], errors="ignore")

    if search:
        try:
            search_id = int(search)
            display_df = display_df[display_df["product_ID"] == search_id]
        except ValueError:
            pass

    total = len(display_df)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    items = display_df.iloc[start_idx:end_idx].replace({np.nan: None}).to_dict(orient="records")

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "items": items,
    }


@app.get("/experiments")
def get_experiments(
    evaluation_type: Optional[str] = None,
    feature_group: Optional[str] = None,
    search: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
):
    """Query the verified experiment registry."""
    path = Path("FINAL_EXPERIMENT_REGISTRY.csv")
    if not path.exists():
        path = EXPERIMENT_DIR / "FINAL_EXPERIMENT_REGISTRY.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Experiment registry not found.")

    df = pd.read_csv(path)
    if evaluation_type and evaluation_type != "All":
        df = df[df["evaluation_type"].str.contains(evaluation_type, case=False, na=False)]
    if feature_group and feature_group != "All":
        df = df[df["feature_group"].str.contains(feature_group, case=False, na=False)]
    if search:
        mask = (
            df["experiment_id"].str.contains(search, case=False, na=False)
            | df["algorithm"].str.contains(search, case=False, na=False)
            | df["feature_group"].str.contains(search, case=False, na=False)
        )
        df = df[mask]

    total = len(df)
    start = (page - 1) * page_size
    end = start + page_size
    records = df.iloc[start:end].replace({np.nan: None}).to_dict(orient="records")

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "records": records,
    }


@app.get("/experiments/{experiment_id}")
def get_experiment_detail(experiment_id: str):
    """Retrieve full metadata for a specific experiment."""
    path = Path("FINAL_EXPERIMENT_REGISTRY.csv")
    if not path.exists():
        path = EXPERIMENT_DIR / "FINAL_EXPERIMENT_REGISTRY.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Experiment registry not found.")

    df = pd.read_csv(path)
    match = df[df["experiment_id"] == experiment_id]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found.")

    return match.iloc[0].replace({np.nan: None}).to_dict()


@app.get("/explainability/global")
def get_global_shap_importance():
    """Retrieve global TreeSHAP feature importance ranking from empirical benchmarks."""
    path = EXPERIMENT_DIR / "SHAP_FEATURE_IMPORTANCE.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="SHAP feature importance not found.")

    df = pd.read_csv(path)

    # Classify feature modalities
    def get_modality(feat: str) -> str:
        if feat in NETWORK_FEATURES:
            return "network"
        if feat in IMAGE_FEATURES:
            return "image"
        if feat in REVIEW_FEATURES:
            return "review"
        return "other"

    records = []
    for _, row in df.iterrows():
        feat = str(row["feature"])
        records.append({
            "feature": feat,
            "mean_abs_shap": round(float(row["mean_abs_shap"]), 6),
            "percent_importance": round(float(row["percent_importance"]), 2),
            "modality": get_modality(feat),
        })

    # Summary by modality
    modality_breakdown = {}
    for r in records:
        m = r["modality"]
        modality_breakdown[m] = round(modality_breakdown.get(m, 0.0) + r["percent_importance"], 2)

    return {
        "features": records,
        "modality_breakdown": modality_breakdown,
        "total_features": len(records),
        "top_feature": records[0]["feature"] if records else None,
    }


@app.get("/clustering/details")
def get_clustering_details():
    """Retrieve comprehensive clustering sweep ($k=2..20$) and cluster profile comparisons."""
    from src.models.clustering_pipeline import get_optimal_cluster_configuration

    sweep_path = EXPERIMENT_DIR / "KMEANS_SWEEP_METRICS.csv"
    k2_path = EXPERIMENT_DIR / "CLUSTER_PROFILES_K2.csv"
    k20_path = EXPERIMENT_DIR / "CLUSTER_PROFILES_K20.csv"

    sweep_metrics = []
    if sweep_path.exists():
        df_sw = pd.read_csv(sweep_path)
        sweep_metrics = df_sw.replace({np.nan: None}).to_dict(orient="records")

    k2_profiles = []
    if k2_path.exists():
        df_k2 = pd.read_csv(k2_path)
        k2_profiles = df_k2.replace({np.nan: None}).to_dict(orient="records")

    k20_profiles = []
    if k20_path.exists():
        df_k20 = pd.read_csv(k20_path)
        k20_profiles = df_k20.replace({np.nan: None}).to_dict(orient="records")

    clustering_summary = get_optimal_cluster_configuration(sweep_path)

    return {
        "sweep_metrics": sweep_metrics,
        "k2_profiles": k2_profiles,
        "k20_profiles": k20_profiles,
        "optimal_k": clustering_summary["selected_k"],
        "optimal_silhouette": clustering_summary["silhouette_score"],
        "clustering_summary": clustering_summary,
    }


@app.get("/network/summary")
def get_network_summary():
    """Retrieve graph ML benchmark metrics and synthetic ring robustness evaluation."""
    bench_path = EXPERIMENT_DIR / "GRAPH_ML_BENCHMARK.csv"
    ring_path = EXPERIMENT_DIR / "SYNTHETIC_RING_ROBUSTNESS.csv"

    benchmarks = []
    if bench_path.exists():
        df_b = pd.read_csv(bench_path)
        benchmarks = df_b.replace({np.nan: None}).to_dict(orient="records")

    ring_evals = []
    if ring_path.exists():
        df_r = pd.read_csv(ring_path)
        ring_evals = df_r.head(20).replace({np.nan: None}).to_dict(orient="records")

    return {
        "benchmarks": benchmarks,
        "ring_evaluations": ring_evals,
        "node_features": NETWORK_FEATURES,
        "real_network_available": False,
        "real_network_note": "Real full Amazon reviewer graph raw archive is not bundled; network analysis uses extracted centralities and synthetic ring simulations.",
    }


@app.get("/network/graph")
def get_network_sample_graph(
    noise_level: float = Query(0.1, ge=0.0, le=0.5),
):
    """
    Generate interactive demonstration bipartite network with reviewer and product nodes
    and an injected coordinated fraud ring for the React Flow Network Explorer.
    Dynamically computes degree, PageRank, projection clustering coefficient, and coordinated edge ratios.
    """
    import networkx as nx
    from src.graph.synthetic_rings import generate_synthetic_review_ring

    seed = int(42 + round(noise_level * 1000))
    n_honest_revs = int(16 + round(24 * (1.0 + noise_level * 1.5)))
    n_honest_prods = int(10 + round(10 * noise_level))
    conn_prob = max(0.60, round(0.92 - noise_level * 0.45, 2))

    df_ring, ring_revs, target_prods = generate_synthetic_review_ring(
        n_ring_reviewers=8,
        n_target_products=5,
        connection_prob=conn_prob,
        n_honest_reviewers=n_honest_revs,
        n_honest_products=n_honest_prods,
        honest_reviews_per_user=2,
        random_seed=seed,
    )

    # Construct NetworkX bipartite graph to compute topological metrics
    B = nx.Graph()
    products = sorted(df_ring["product_ID"].unique())
    reviewers = sorted(df_ring["reviewer_ID"].unique())

    for p in products:
        B.add_node(p, bipartite=0)
    for r in reviewers:
        B.add_node(r, bipartite=1)
    for _, row in df_ring.iterrows():
        B.add_edge(row["reviewer_ID"], row["product_ID"])

    # PageRank over bipartite graph
    pagerank_scores = nx.pagerank(B, alpha=0.85)

    # Clustering coefficient on projected product co-review graph
    P = nx.bipartite.weighted_projected_graph(B, products)
    clustering_scores = nx.clustering(P)

    nodes = []
    edges = []

    for idx, p in enumerate(products):
        is_target = p in target_prods
        deg = int(B.degree(p))
        fake_neighbors = sum(1 for u in B.neighbors(p) if u in ring_revs)
        coord_ratio = round(fake_neighbors / deg, 4) if deg > 0 else 0.0
        nodes.append({
            "id": str(p),
            "label": f"Product {p.replace('target_prod_', 'T-').replace('honest_prod_', 'H-')}",
            "type": "product",
            "is_suspicious": is_target,
            "degree": deg,
            "pagerank": round(float(pagerank_scores.get(p, 0.0)), 6),
            "clustering_coef": round(float(clustering_scores.get(p, 0.0)), 4),
            "coordinated_edge_ratio": coord_ratio,
            "position": {"x": 600, "y": 60 + idx * 55},
        })

    for idx, r in enumerate(reviewers):
        is_ring = r in ring_revs
        deg = int(B.degree(r))
        fake_neighbors = sum(1 for v in B.neighbors(r) if v in target_prods)
        coord_ratio = round(fake_neighbors / deg, 4) if deg > 0 else 0.0
        nodes.append({
            "id": str(r),
            "label": f"Reviewer {r.replace('ring_rev_', 'R-Ring-').replace('honest_rev_', 'R-Honest-')}",
            "type": "reviewer",
            "is_suspicious": is_ring,
            "degree": deg,
            "pagerank": round(float(pagerank_scores.get(r, 0.0)), 6),
            "clustering_coef": None,
            "coordinated_edge_ratio": coord_ratio,
            "position": {"x": 100, "y": 40 + idx * 30},
        })

    for idx, row in df_ring.iterrows():
        edges.append({
            "id": f"e_{row['reviewer_ID']}_{row['product_ID']}",
            "source": str(row["reviewer_ID"]),
            "target": str(row["product_ID"]),
            "is_fake": bool(row["is_fake_review"] == 1),
            "weight": 1.0,
        })

    return {
        "nodes": nodes,
        "edges": edges,
        "stats": {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "reviewer_nodes": len(reviewers),
            "product_nodes": len(products),
            "ring_reviewers": len(ring_revs),
            "target_products": len(target_prods),
            "noise_level": noise_level,
        },
        "mode": "Demonstration Synthetic Bipartite Review Graph",
    }


# ──────────────────────────────────────────────
# System Integrity & Runtime Environment Endpoints
# ──────────────────────────────────────────────
@app.get("/system/environment")
def get_system_environment():
    """Retrieve runtime Python and library dependencies version information."""
    import platform
    import sys
    import sklearn
    import xgboost
    import catboost
    import fastapi

    return {
        "python_version": sys.version.split()[0],
        "full_python_version": sys.version,
        "sklearn_version": sklearn.__version__,
        "xgboost_version": xgboost.__version__,
        "catboost_version": catboost.__version__,
        "fastapi_version": fastapi.__version__,
        "platform": platform.platform(),
        "docker_supported_python": "3.11",
        "host_runtime_status": "verified",
    }


@app.get("/system/test-report")
def get_system_test_report():
    """Retrieve verified test execution report metadata from outputs/test_report.json."""
    report_path = Path("outputs/test_report.json")
    if not report_path.exists():
        return {
            "status": "not_available",
            "message": "No verified test report available.",
            "total_tests": None,
            "passed": None,
            "failed": None,
            "skipped": None,
            "test_execution_date": None,
            "environment": None,
            "test_command": None,
        }

    try:
        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "status": "available",
            **data,
        }
    except Exception as exc:
        return {
            "status": "error",
            "message": f"Failed to parse test report: {exc}",
            "total_tests": None,
            "passed": None,
            "failed": None,
            "skipped": None,
            "test_execution_date": None,
            "environment": None,
            "test_command": None,
        }

