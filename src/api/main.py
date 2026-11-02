"""
Production-grade FastAPI Backend for Fake Review Detection.

Provides inference, explainability (EN, HI, Hinglish), clustering summaries,
and benchmark registry inspection.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import time

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np

from src.config import (
    ALL_STRUCTURED_FEATURES,
    EXPERIMENT_DIR,
    MODEL_DIR,
    REVIEW_NETWORK_FEATURES,
    TARGET,
)
from src.evaluation.risk_score import compute_risk_score, compute_trust_adjusted_rating
from src.evaluation.explainability import explain_single_product
from src.data.data_loader import load_product_data

logger = logging.getLogger("fastapi_app")

app = FastAPI(
    title="Fake Review Detection API",
    description="Explainable Graph ML and Classification API for detecting fake review buyers on Amazon.",
    version="1.0.0",
)

# Enable CORS for frontend integration (e.g. Lovable / Streamlit)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory model cache
_MODELS: Dict[str, Any] = {}
_DF_LABELED: Optional[pd.DataFrame] = None


def get_cached_model(name: str = "review_network_image_xgboost"):
    """Load or retrieve cached trained scikit-learn / XGBoost pipeline."""
    if name not in _MODELS:
        path = MODEL_DIR / f"{name}_pipeline.joblib"
        if not path.exists():
            # Try alternate tuned model
            path = MODEL_DIR / f"tuned_xgboost_pipeline.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found at {path}")
        logger.info("Loading model artifact from %s", path)
        _MODELS[name] = joblib.load(path)
    return _MODELS[name]


def get_labeled_data() -> pd.DataFrame:
    global _DF_LABELED
    if _DF_LABELED is None:
        _DF_LABELED = load_product_data(with_images=True)
    return _DF_LABELED


# ──────────────────────────────────────────────
# Pydantic Request / Response Models
# ──────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str


class ProductFeatureInput(BaseModel):
    product_ID: Optional[int] = None
    features: Dict[str, float]
    model_name: str = "review_network_image_xgboost"
    displayed_rating: float = 4.5


class PredictionResponse(BaseModel):
    product_ID: Optional[int]
    model_name: str
    p_fake: float
    is_fake_prediction: bool
    risk_score: float
    risk_tier: str
    trust_adjusted_rating: float
    explanation_english: str
    explanation_hindi: str
    explanation_hinglish: str


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
def health():
    return {
        "status": "ok",
        "version": "1.0.0",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


@app.get("/datasets/summary")
def get_datasets_summary():
    df = get_labeled_data()
    return {
        "labeled_products": len(df),
        "features_count": len(df.columns) - 2,
        "positive_class_count": int(df[TARGET].sum()),
        "negative_class_count": int((df[TARGET] == 0).sum()),
        "fake_prevalence_pct": round(float(df[TARGET].mean()) * 100.0, 2),
        "available_feature_groups": [
            "review_features (12)",
            "network_features (4)",
            "image_features (12)",
            "text_tfidf (1000)",
        ],
    }


@app.get("/models/metrics")
def get_metrics(
    model_name: Optional[str] = None,
    feature_group: Optional[str] = None,
):
    results_path = EXPERIMENT_DIR / "EXPERIMENT_RESULTS.csv"
    if not results_path.exists():
        raise HTTPException(status_code=404, detail="No experiment results registered yet.")

    df_res = pd.read_csv(results_path)
    if model_name:
        df_res = df_res[df_res["model"].str.contains(model_name, case=False, na=False)]
    if feature_group:
        df_res = df_res[df_res["feature_group"].str.contains(feature_group, case=False, na=False)]

    return df_res.to_dict(orient="records")


@app.post("/predict/product", response_model=PredictionResponse)
def predict_product(payload: ProductFeatureInput):
    try:
        pipeline = get_cached_model(payload.model_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Construct DataFrame in expected column order
    input_data = pd.DataFrame([payload.features])

    # Fill any missing expected features with 0.0
    for col in ALL_STRUCTURED_FEATURES:
        if col not in input_data.columns:
            input_data[col] = 0.0
    input_data = input_data[ALL_STRUCTURED_FEATURES]

    # Model inference
    try:
        p_fake = float(pipeline.predict_proba(input_data)[0, 1])
    except Exception:
        p_fake = float(pipeline.predict(input_data)[0])

    is_fake = p_fake >= 0.5
    risk_score, risk_tier = compute_risk_score(p_fake)
    adjusted_rating = compute_trust_adjusted_rating(payload.displayed_rating, p_fake)

    # Localized explanations
    model_raw = pipeline.named_steps["classifier"] if hasattr(pipeline, "named_steps") else pipeline
    try:
        import shap
        explainer = shap.TreeExplainer(model_raw)
        exp = explain_single_product(explainer, input_data, ALL_STRUCTURED_FEATURES)
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
        "p_fake": round(p_fake, 4),
        "is_fake_prediction": is_fake,
        "risk_score": risk_score,
        "risk_tier": risk_tier,
        "trust_adjusted_rating": adjusted_rating,
        "explanation_english": exp_en,
        "explanation_hindi": exp_hi,
        "explanation_hinglish": exp_hing,
    }


@app.get("/products/{product_id}")
def get_product_details(product_id: int):
    df = get_labeled_data()
    match = df[df["product_ID"] == product_id]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"Product {product_id} not found in database.")

    row = match.iloc[0].to_dict()
    target_val = int(row.pop(TARGET)) if TARGET in row else None
    return {
        "product_ID": product_id,
        "ground_truth_label": "FAKE_BUYER" if target_val == 1 else "GENUINE",
        "ground_truth_binary": target_val,
        "features": {k: float(v) for k, v in row.items() if k != "product_ID"},
    }


@app.get("/clustering/summary")
def get_clustering_summary():
    path = EXPERIMENT_DIR / "CLUSTER_ANALYSIS_SUMMARY.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Cluster summary not generated yet.")
    df_c = pd.read_csv(path)
    return df_c.to_dict(orient="records")
