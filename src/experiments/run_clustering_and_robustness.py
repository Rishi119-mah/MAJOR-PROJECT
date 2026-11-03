"""
Run Clustering, Anomaly Detection, and Synthetic Review Ring Robustness Experiments.

RIGOROUS EVALUATION PROTOCOL:
1. K-Means sweep (k=2..20) with Silhouette, Davies-Bouldin, and Calinski-Harabasz metrics.
2. Separate cluster profiling for k=2 (mathematical silhouette peak) and k=20 (paper default).
3. Strict separation between:
   - Ground-truth fake prevalence (actual observed binary labels).
   - Ground-truth genuine prevalence (1 - fake prevalence).
   - Unbiased out-of-fold predicted fake probabilities (via 5-fold CV).
4. Synthetic review ring robustness evaluating both Recall AND Precision, Specificity, FPR.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import EXPERIMENT_DIR, REVIEW_NETWORK_FEATURES, TARGET, RANDOM_SEED
from src.data.data_loader import load_product_data
from src.models.clustering_pipeline import (
    evaluate_kmeans_sweep,
    analyze_clusters,
    run_isolation_forest_anomaly,
)
from src.graph.synthetic_rings import evaluate_ring_robustness

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("clustering_robustness")


def compute_unbiased_oof_predictions(df: pd.DataFrame, features: list[str]) -> np.ndarray:
    """Compute out-of-fold predicted fake probabilities via 5-fold CV to prevent training bias."""
    X = df[features].values
    y = df[TARGET].values
    oof_probs = np.zeros(len(df))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    for tr_idx, val_idx in skf.split(X, y):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr_idx])
        X_val = scaler.transform(X[val_idx])

        clf = xgb.XGBClassifier(eval_metric="logloss", random_state=RANDOM_SEED, n_estimators=100, max_depth=4)
        clf.fit(X_tr, y[tr_idx])
        oof_probs[val_idx] = clf.predict_proba(X_val)[:, 1]

    return oof_probs


def main():
    logger.info("Loading labelled product dataset...")
    df_lab = load_product_data(with_images=False)

    # 1. K-Means Sweep (k=2..20)
    logger.info("Running K-Means sweep on labelled dataset (k=2..20)...")
    df_sweep = evaluate_kmeans_sweep(df_lab, features=REVIEW_NETWORK_FEATURES, k_min=2, k_max=20)
    best_sil = df_sweep.loc[df_sweep["silhouette"].idxmax()]
    print("\n" + "=" * 70)
    print(f"OPTIMAL K-MEANS CLUSTER COUNT (SILHOUETTE): k={int(best_sil['k'])} (Score: {best_sil['silhouette']:.4f})")
    print(f"K=20 (Paper Default) Silhouette Score: {df_sweep[df_sweep['k']==20]['silhouette'].values[0]:.4f}")
    print("=" * 70)

    # 2. Compute Unbiased Out-Of-Fold Predicted Probabilities
    logger.info("Generating unbiased out-of-fold fake probability predictions...")
    oof_p_fake = compute_unbiased_oof_predictions(df_lab, REVIEW_NETWORK_FEATURES)

    # 3. Cluster Profiling for k=2 (Optimal Silhouette)
    logger.info("Generating Cluster Profile for k=2...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_lab[REVIEW_NETWORK_FEATURES])
    km2 = KMeans(n_clusters=2, random_state=RANDOM_SEED, n_init=10).fit(X_scaled)

    df_k2 = analyze_clusters(
        df_lab,
        cluster_labels=km2.labels_,
        features=REVIEW_NETWORK_FEATURES,
        p_fake_preds=oof_p_fake,
    )
    df_k2["ground_truth_genuine_prevalence"] = 1.0 - df_k2["ground_truth_fake_prevalence"]
    df_k2.to_csv(EXPERIMENT_DIR / "CLUSTER_PROFILES_K2.csv", index=False)
    print("\nCLUSTER PROFILE FOR k=2 (Optimal Silhouette):")
    cols_k2 = ["cluster_ID", "n_products", "share_of_total", "ground_truth_fake_prevalence", "ground_truth_genuine_prevalence", "mean_p_fake_predicted"]
    print(df_k2[cols_k2].to_string(index=False))

    # 4. Cluster Profiling for k=20 (Paper Default)
    logger.info("Generating Cluster Profile for k=20...")
    km20 = KMeans(n_clusters=20, random_state=RANDOM_SEED, n_init=10).fit(X_scaled)
    df_k20 = analyze_clusters(
        df_lab,
        cluster_labels=km20.labels_,
        features=REVIEW_NETWORK_FEATURES,
        p_fake_preds=oof_p_fake,
    )
    df_k20["ground_truth_genuine_prevalence"] = 1.0 - df_k20["ground_truth_fake_prevalence"]
    df_k20.to_csv(EXPERIMENT_DIR / "CLUSTER_PROFILES_K20.csv", index=False)

    # Also save as main summary
    df_k20.to_csv(EXPERIMENT_DIR / "CLUSTER_ANALYSIS_SUMMARY.csv", index=False)
    print("\nCLUSTER PROFILE FOR k=20 (Paper Default - Top 5 Clusters by Size):")
    print(df_k20[cols_k2].head(5).to_string(index=False))

    # 5. Isolation Forest Anomaly Detection
    logger.info("Running Isolation Forest Anomaly Detection...")
    df_anom = run_isolation_forest_anomaly(df_lab, features=REVIEW_NETWORK_FEATURES, contamination=0.10)
    anom_fake_rate = df_anom[df_anom["is_anomaly"] == 1][TARGET].mean()
    norm_fake_rate = df_anom[df_anom["is_anomaly"] == 0][TARGET].mean()
    print("\n" + "=" * 70)
    print(f"ISOLATION FOREST ANOMALIES: Fake Rate in Anomalous Cohort = {anom_fake_rate*100:.2f}% vs Normal Cohort = {norm_fake_rate*100:.2f}%")
    print("=" * 70)

    # 6. Synthetic Fraud Ring Robustness Evaluation (Evaluating Recall AND Precision)
    logger.info("Running Synthetic Fraud Ring Robustness Evaluation...")
    df_robust = evaluate_ring_robustness(
        noise_levels=[0.0, 0.1, 0.2, 0.3, 0.5],
        ring_sizes=[5, 10, 20, 30, 50],
        random_seeds=[42, 123],
    )
    df_robust.to_csv(EXPERIMENT_DIR / "SYNTHETIC_RING_ROBUSTNESS.csv", index=False)
    print("\nSYNTHETIC FRAUD RING ROBUSTNESS SUMMARY (Sample across configurations):")
    summary_cols = ["heuristic_strategy", "ring_size", "noise_level", "recall", "precision", "f1_score", "false_positive_rate"]
    print(df_robust[summary_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
