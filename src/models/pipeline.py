"""
Pipeline construction, experiment execution, and cross-validation runners.

Ensures strict separation between legacy baseline reproduction and
clean, leakage-safe improved experiments.
"""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    CV_REPEATS,
    CV_SPLITS,
    EXPERIMENT_DIR,
    RANDOM_SEED,
    TARGET,
    TEST_SIZE,
)
from src.evaluation.metrics import compute_all_metrics
from src.models.classifiers import get_legacy_model, get_model

logger = logging.getLogger(__name__)


def build_pipeline(
    model: BaseEstimator,
    scale: bool = True,
) -> Pipeline:
    """Build a scikit-learn Pipeline with optional scaling.

    Parameters
    ----------
    model : BaseEstimator
        The classifier instance.
    scale : bool, default True
        Whether to prepend a StandardScaler step.

    Returns
    -------
    Pipeline
        Configured sklearn pipeline.
    """
    steps = []
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("classifier", model))
    return Pipeline(steps)


def run_legacy_experiment(
    df: pd.DataFrame,
    features: list[str] | None = None,
    experiment_name: str = "legacy_experiment",
) -> dict[str, dict[str, Any]]:
    """Reproduce the EXACT behavior of the original classification.py.

    - Uses single train_test_split(test_size=0.20, random_state=42) WITHOUT stratification.
    - Applies StandardScaler to ALL models (including tree models).
    - Runs LogisticRegression, RandomForest, SVC(linear), and XGBoost.
    - Returns metrics matching the original paper's reporting.
    """
    if features is None:
        X = df.drop(columns=["product_ID", TARGET], errors="ignore")
        features = list(X.columns)
        y = df[TARGET]
    else:
        X = df[features]
        y = df[TARGET]

    # Non-stratified split as in original code
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    legacy_model_names = ["logistic_regression", "random_forest", "svm_linear", "xgboost"]
    results = {}

    for m_name in legacy_model_names:
        model = get_legacy_model(m_name)
        t0 = time.perf_counter()
        model.fit(X_train_scaled, y_train)
        train_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        y_pred = model.predict(X_test_scaled)
        inference_time = time.perf_counter() - t1

        y_prob = None
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        elif hasattr(model, "decision_function"):
            dfun = model.decision_function(X_test_scaled)
            y_prob = 1 / (1 + np.exp(-dfun))

        metrics = compute_all_metrics(y_test, y_pred, y_prob)
        metrics["train_time_s"] = train_time
        metrics["inference_time_s"] = inference_time
        metrics["n_train"] = len(X_train)
        metrics["n_test"] = len(X_test)
        metrics["experiment_name"] = experiment_name
        metrics["model_name"] = m_name
        metrics["is_legacy"] = True

        results[m_name] = metrics

    return results


def run_single_experiment(
    df: pd.DataFrame,
    features: list[str],
    model_name: str,
    *,
    stratify: bool = True,
    scale: bool | None = None,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_SEED,
    experiment_id: str = "exp_single",
    **model_kwargs: Any,
) -> dict[str, Any]:
    """Execute a single leakage-safe model evaluation on a holdout split.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing features and TARGET.
    features : list[str]
        Feature subset to use.
    model_name : str
        Model name passed to get_model.
    stratify : bool, default True
        Whether to use stratified split on TARGET.
    scale : bool | None
        Whether to scale features. If None, automatically determined based on model.
    test_size : float
        Holdout fraction.
    random_state : int
        Seed for reproducibility.
    experiment_id : str
        Tracking identifier.

    Returns
    -------
    dict[str, Any]
        Dictionary of metrics, model metadata, and timings.
    """
    X = df[features].copy()
    y = df[TARGET].copy()

    strat = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=strat
    )

    # Determine whether to scale
    if scale is None:
        needs_scale = ["dummy", "logistic_regression", "svm_linear", "svm_rbf", "stacking"]
        scale = model_name.lower() in needs_scale

    raw_model = get_model(model_name, **model_kwargs)
    pipeline = build_pipeline(raw_model, scale=scale)

    t0 = time.perf_counter()
    pipeline.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    y_pred = pipeline.predict(X_test)
    inference_time = time.perf_counter() - t1

    y_prob = None
    if hasattr(pipeline, "predict_proba"):
        y_prob = pipeline.predict_proba(X_test)[:, 1]
    elif hasattr(pipeline, "decision_function"):
        dfun = pipeline.decision_function(X_test)
        y_prob = 1 / (1 + np.exp(-dfun))

    metrics = compute_all_metrics(y_test, y_pred, y_prob)
    metrics["experiment_id"] = experiment_id
    metrics["model_name"] = model_name
    metrics["feature_group"] = experiment_id
    metrics["train_time_s"] = train_time
    metrics["inference_time_s"] = inference_time
    metrics["n_train"] = len(X_train)
    metrics["n_test"] = len(X_test)
    metrics["pipeline"] = pipeline
    metrics["y_test"] = y_test.values
    metrics["y_pred"] = y_pred
    metrics["y_prob"] = y_prob

    return metrics


def run_cv_experiment(
    df: pd.DataFrame,
    features: list[str],
    model_name: str,
    *,
    n_splits: int = CV_SPLITS,
    n_repeats: int = CV_REPEATS,
    scale: bool | None = None,
    random_state: int = RANDOM_SEED,
    experiment_id: str = "exp_cv",
    **model_kwargs: Any,
) -> dict[str, Any]:
    """Execute repeated stratified cross-validation.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset.
    features : list[str]
        List of feature column names.
    model_name : str
        Model to evaluate.
    n_splits : int
        Number of folds per repeat.
    n_repeats : int
        Number of CV repeats.
    scale : bool | None
        Whether to scale inside each fold.
    random_state : int
        RNG seed.
    experiment_id : str
        Identifier.

    Returns
    -------
    dict[str, Any]
        Mean and standard deviation of each metric across all folds.
    """
    X = df[features].copy()
    y = df[TARGET].copy()

    if scale is None:
        needs_scale = ["dummy", "logistic_regression", "svm_linear", "svm_rbf", "stacking"]
        scale = model_name.lower() in needs_scale

    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )

    fold_metrics: list[dict[str, Any]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(rskf.split(X, y)):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        model = get_model(model_name, **model_kwargs)
        pipeline = build_pipeline(model, scale=scale)

        t0 = time.perf_counter()
        pipeline.fit(X_train_fold, y_train_fold)
        train_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        y_pred = pipeline.predict(X_val_fold)
        inf_time = time.perf_counter() - t1

        y_prob = None
        if hasattr(pipeline, "predict_proba"):
            y_prob = pipeline.predict_proba(X_val_fold)[:, 1]
        elif hasattr(pipeline, "decision_function"):
            dfun = pipeline.decision_function(X_val_fold)
            y_prob = 1 / (1 + np.exp(-dfun))

        m = compute_all_metrics(y_val_fold, y_pred, y_prob)
        m["train_time_s"] = train_time
        m["inference_time_s"] = inf_time
        fold_metrics.append(m)

    # Compute mean and std for numeric metrics
    metric_keys = [
        "accuracy",
        "fake_precision",
        "fake_recall",
        "fake_f1",
        "genuine_precision",
        "genuine_recall",
        "genuine_f1",
        "macro_f1",
        "weighted_f1",
        "specificity",
        "roc_auc",
        "pr_auc",
        "brier_score",
        "train_time_s",
        "inference_time_s",
    ]

    summary: dict[str, Any] = {
        "experiment_id": experiment_id,
        "model_name": model_name,
        "n_folds": n_splits * n_repeats,
    }

    for k in metric_keys:
        vals = [f[k] for f in fold_metrics if f[k] is not None]
        if vals:
            summary[f"{k}_mean"] = float(np.mean(vals))
            summary[f"{k}_std"] = float(np.std(vals))
        else:
            summary[f"{k}_mean"] = None
            summary[f"{k}_std"] = None

    summary["fold_metrics"] = fold_metrics
    return summary


def log_experiment_result(
    experiment_id: str,
    model: str,
    feature_group: str,
    metrics: dict[str, Any],
    results_csv: Path | None = None,
) -> None:
    """Append a single experiment result row to EXPERIMENT_RESULTS.csv."""
    if results_csv is None:
        results_csv = EXPERIMENT_DIR / "EXPERIMENT_RESULTS.csv"

    row = {
        "experiment_id": experiment_id,
        "model": model,
        "feature_group": feature_group,
        "accuracy": f"{metrics.get('accuracy', 0.0):.4f}",
        "fake_precision": f"{metrics.get('fake_precision', 0.0):.4f}",
        "fake_recall": f"{metrics.get('fake_recall', 0.0):.4f}",
        "fake_f1": f"{metrics.get('fake_f1', 0.0):.4f}",
        "macro_f1": f"{metrics.get('macro_f1', 0.0):.4f}",
        "weighted_f1": f"{metrics.get('weighted_f1', 0.0):.4f}",
        "roc_auc": (
            f"{metrics.get('roc_auc'):.4f}" if metrics.get("roc_auc") is not None else ""
        ),
        "pr_auc": (
            f"{metrics.get('pr_auc'):.4f}" if metrics.get("pr_auc") is not None else ""
        ),
        "brier_score": (
            f"{metrics.get('brier_score'):.4f}"
            if metrics.get("brier_score") is not None
            else ""
        ),
        "specificity": f"{metrics.get('specificity', 0.0):.4f}",
        "sensitivity": f"{metrics.get('fake_recall', 0.0):.4f}",
        "train_time_s": f"{metrics.get('train_time_s', 0.0):.4f}",
        "inference_time_s": f"{metrics.get('inference_time_s', 0.0):.4f}",
        "n_train": metrics.get("n_train", 0),
        "n_test": metrics.get("n_test", 0),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    file_exists = results_csv.exists()
    fieldnames = list(row.keys())

    with open(results_csv, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists or results_csv.stat().st_size == 0:
            writer.writeheader()
        writer.writerow(row)

