"""
Run Clustering, Anomaly Detection, and Synthetic Review Ring Robustness Experiments.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import EXPERIMENT_DIR, REVIEW_NETWORK_FEATURES, TARGET
from src.data.data_loader import load_product_data, load_ucsd_data
from src.models.clustering_pipeline import (
    evaluate_kmeans_sweep,
    analyze_clusters,
    run_isolation_forest_anomaly,
)
from src.graph.synthetic_rings import evaluate_ring_robustness
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("clustering_robustness")


def main():
    logger.info("Loading labelled product dataset...")
    df_lab = load_product_data(with_images=False)

    # 1. K-Means Sweep (k=2..20) on labelled dataset
    logger.info("Running K-Means sweep on labelled dataset...")
    df_sweep = evaluate_kmeans_sweep(df_lab, features=REVIEW_NETWORK_FEATURES, k_min=2, k_max=20)
    best_sil = df_sweep.loc[df_sweep["silhouette"].idxmax()]
    print("\n" + "=" * 60)
    print(f"OPTIMAL K-MEANS CLUSTER COUNT BY SILHOUETTE: k={int(best_sil['k'])} (Silhouette={best_sil['silhouette']:.4f})")
    print(f"K=20 (Paper default) Silhouette: {df_sweep[df_sweep['k']==20]['silhouette'].values[0]:.4f}")
    print("=" * 60)

    # 2. Cluster Analysis for k=20 (to match paper) with ground truth vs predicted
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    X_scaled = StandardScaler().fit_transform(df_lab[REVIEW_NETWORK_FEATURES])
    km20 = KMeans(n_clusters=20, random_state=42, n_init=10).fit(X_scaled)

    # Predict fake probability using our trained Random Forest model
    rf_path = project_root / "outputs" / "models" / "review_network_random_forest_pipeline.joblib"
    if rf_path.exists():
        pipe = joblib.load(rf_path)
        p_fake = pipe.predict_proba(df_lab[REVIEW_NETWORK_FEATURES])[:, 1]
    else:
        p_fake = None

    df_cluster_summary = analyze_clusters(
        df_lab,
        cluster_labels=km20.labels_,
        features=REVIEW_NETWORK_FEATURES,
        p_fake_preds=p_fake,
    )
    print("\nCLUSTER ANALYSIS SUMMARY (Top 5 clusters by size):")
    cols_to_show = ["cluster_ID", "n_products", "share_of_total", "ground_truth_fake_prevalence"]
    if p_fake is not None:
        cols_to_show.append("mean_p_fake_predicted")
    print(df_cluster_summary[cols_to_show].head(5).to_string(index=False))

    # 3. Isolation Forest Anomaly Detection
    logger.info("Running Isolation Forest Anomaly Detection...")
    df_anom = run_isolation_forest_anomaly(df_lab, features=REVIEW_NETWORK_FEATURES, contamination=0.10)
    anom_fake_rate = df_anom[df_anom["is_anomaly"] == 1][TARGET].mean()
    norm_fake_rate = df_anom[df_anom["is_anomaly"] == 0][TARGET].mean()
    print("\n" + "=" * 60)
    print(f"ISOLATION FOREST ANOMALIES: Fake Rate in Anomalous Cohort = {anom_fake_rate*100:.2f}% vs Normal Cohort = {norm_fake_rate*100:.2f}%")
    print("=" * 60)

    # 4. Synthetic Fraud Ring Robustness Evaluation
    logger.info("Running Synthetic Fraud Ring Robustness Evaluation...")
    from src.graph.synthetic_rings import evaluate_ring_robustness
    df_robust = evaluate_ring_robustness(
        noise_levels=[0.0, 0.2, 0.4],
        ring_sizes=[5, 15, 30],
    )
    df_robust.to_csv(EXPERIMENT_DIR / "SYNTHETIC_RING_ROBUSTNESS.csv", index=False)
    print("\nSYNTHETIC FRAUD RING ROBUSTNESS SUMMARY:")
    print(df_robust[["ring_size", "noise_level", "recall", "precision", "f1_score"]].to_string(index=False))


if __name__ == "__main__":
    main()
