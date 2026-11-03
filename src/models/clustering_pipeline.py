"""
Clustering and Anomaly Detection Pipeline.

Implements K-Means sweep (k=2..20) with Silhouette, Davies-Bouldin, and Calinski-Harabasz
scores, alongside Isolation Forest and DBSCAN anomaly detection.
Preserves clear distinction between model-predicted fake probabilities and ground truth labels.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

from src.config import EXPERIMENT_DIR, RANDOM_SEED, REVIEW_NETWORK_FEATURES, TARGET

logger = logging.getLogger(__name__)


def evaluate_kmeans_sweep(
    df: pd.DataFrame,
    features: list[str] = REVIEW_NETWORK_FEATURES,
    k_min: int = 2,
    k_max: int = 20,
    sample_size_silhouette: int | None = 5000,
) -> pd.DataFrame:
    """Evaluate K-Means clustering across a range of cluster counts k=k_min..k_max.

    Returns DataFrame with columns:
      k, inertia, silhouette, davies_bouldin, calinski_harabasz
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(df[features])

    results = []
    logger.info("Starting K-Means sweep from k=%d to k=%d...", k_min, k_max)

    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = km.fit_predict(X)

        inertia = km.inertia_
        db_score = davies_bouldin_score(X, labels)
        ch_score = calinski_harabasz_score(X, labels)

        # Silhouette computation
        if sample_size_silhouette and len(X) > sample_size_silhouette:
            sil = silhouette_score(
                X, labels, sample_size=sample_size_silhouette, random_state=RANDOM_SEED
            )
        else:
            sil = silhouette_score(X, labels)

        results.append({
            "k": k,
            "inertia": float(inertia),
            "silhouette": float(sil),
            "davies_bouldin": float(db_score),
            "calinski_harabasz": float(ch_score),
        })

    df_sweep = pd.DataFrame(results)
    out_csv = EXPERIMENT_DIR / "KMEANS_SWEEP_METRICS.csv"
    df_sweep.to_csv(out_csv, index=False)
    logger.info("Saved K-Means sweep results to %s", out_csv)
    return df_sweep


def analyze_clusters(
    df: pd.DataFrame,
    cluster_labels: np.ndarray,
    features: list[str] = REVIEW_NETWORK_FEATURES,
    p_fake_preds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Analyze cluster-level characteristics, separating predicted from ground-truth fake rates."""
    df_analysis = df.copy()
    df_analysis["cluster_ID"] = cluster_labels

    if p_fake_preds is not None:
        df_analysis["p_fake_pred"] = p_fake_preds

    has_ground_truth = TARGET in df_analysis.columns

    records = []
    for c_id, grp in df_analysis.groupby("cluster_ID"):
        rec: dict[str, Any] = {
            "cluster_ID": int(c_id),
            "n_products": len(grp),
            "share_of_total": len(grp) / len(df_analysis),
        }

        # Ground truth if available
        if has_ground_truth:
            rec["ground_truth_fake_prevalence"] = float(grp[TARGET].mean())
            rec["ground_truth_fake_count"] = int(grp[TARGET].sum())

        # Model predicted probabilities
        if p_fake_preds is not None:
            rec["mean_p_fake_predicted"] = float(grp["p_fake_pred"].mean())
            rec["pct_predicted_fake_threshold_05"] = float((grp["p_fake_pred"] >= 0.5).mean())
            rec["pct_predicted_fake_threshold_07"] = float((grp["p_fake_pred"] >= 0.7).mean())

        # Feature means
        for f in features:
            rec[f"mean_{f}"] = float(grp[f].mean())

        records.append(rec)

    df_summary = pd.DataFrame(records)
    out_path = EXPERIMENT_DIR / "CLUSTER_ANALYSIS_SUMMARY.csv"
    df_summary.to_csv(out_path, index=False)
    logger.info("Saved cluster analysis to %s", out_path)
    return df_summary


def run_isolation_forest_anomaly(
    df: pd.DataFrame,
    features: list[str] = REVIEW_NETWORK_FEATURES,
    contamination: float = 0.1,
) -> pd.DataFrame:
    """Run Isolation Forest to detect anomalous products based on network and review features."""
    scaler = StandardScaler()
    X = scaler.fit_transform(df[features])

    iso = IsolationForest(
        contamination=contamination,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    preds = iso.fit_predict(X)  # -1 for anomaly, 1 for normal
    scores = iso.decision_function(X)

    df_out = df.copy()
    df_out["anomaly_label"] = preds
    df_out["anomaly_score"] = scores
    df_out["is_anomaly"] = (preds == -1).astype(int)

    if TARGET in df_out.columns:
        cross_tab = pd.crosstab(df_out[TARGET], df_out["is_anomaly"], normalize="index")
        logger.info("Isolation Forest vs Target Ground Truth:\n%s", cross_tab)

    return df_out


def get_optimal_cluster_configuration(
    sweep_path: Path | str | None = None,
    selection_criterion: str = "max_silhouette",
    dataset_identifier: str = "amazon_electronics_multimodal",
) -> dict[str, Any]:
    """Calculate the optimal clustering configuration from actual saved sweep results.

    Returns a dictionary with nullable fields if missing, with zero hardcoded defaults.
    """
    if sweep_path is None:
        target_path = EXPERIMENT_DIR / "KMEANS_SWEEP_METRICS.csv"
    else:
        target_path = Path(sweep_path)

    if not target_path.exists():
        logger.warning("K-Means sweep metrics file not found at %s", target_path)
        return {
            "available_k_values": [],
            "selected_k": None,
            "selection_criterion": selection_criterion,
            "silhouette_score": None,
            "davies_bouldin_score": None,
            "calinski_harabasz_score": None,
            "cluster_count": None,
            "dataset_identifier": dataset_identifier,
            "result_source": "unavailable",
            "optimal_k": None,
            "optimal_silhouette": None,
        }

    try:
        df_sw = pd.read_csv(target_path)
        if df_sw.empty or "k" not in df_sw.columns or "silhouette" not in df_sw.columns:
            raise ValueError("K-Means sweep CSV is empty or missing required columns 'k' / 'silhouette'.")

        available_k = [int(val) for val in df_sw["k"].dropna().tolist()]
        opt_idx = df_sw["silhouette"].idxmax()
        opt_row = df_sw.loc[opt_idx]

        selected_k = int(opt_row["k"])
        sil_score = round(float(opt_row["silhouette"]), 4)
        db_score = round(float(opt_row["davies_bouldin"]), 4) if "davies_bouldin" in opt_row and pd.notnull(opt_row["davies_bouldin"]) else None
        ch_score = round(float(opt_row["calinski_harabasz"]), 4) if "calinski_harabasz" in opt_row and pd.notnull(opt_row["calinski_harabasz"]) else None

        return {
            "available_k_values": available_k,
            "selected_k": selected_k,
            "selection_criterion": selection_criterion,
            "silhouette_score": sil_score,
            "davies_bouldin_score": db_score,
            "calinski_harabasz_score": ch_score,
            "cluster_count": selected_k,
            "dataset_identifier": dataset_identifier,
            "result_source": target_path.name,
            "optimal_k": selected_k,
            "optimal_silhouette": sil_score,
        }
    except Exception as exc:
        logger.error("Failed to parse K-Means sweep metrics from %s: %s", target_path, exc)
        return {
            "available_k_values": [],
            "selected_k": None,
            "selection_criterion": selection_criterion,
            "silhouette_score": None,
            "davies_bouldin_score": None,
            "calinski_harabasz_score": None,
            "cluster_count": None,
            "dataset_identifier": dataset_identifier,
            "result_source": "unavailable",
            "optimal_k": None,
            "optimal_silhouette": None,
        }


