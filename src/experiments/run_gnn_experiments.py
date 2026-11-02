"""
Run Graph Machine Learning Experiments:
Handcrafted Network Features vs Spectral/Node2Vec Embeddings vs GraphSAGE Aggregations.
Investigates the PPT claim of GraphSAGE F1=0.822.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
import xgboost as xgb

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import EXPERIMENT_DIR, NETWORK_FEATURES, REVIEW_FEATURES, TARGET
from src.data.data_loader import load_product_data
from src.graph.gnn_models import compute_spectral_graph_embeddings, compute_graphsage_aggregations
from src.models.pipeline import log_experiment_result

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("gnn_benchmark")


def main():
    logger.info("Loading labelled product dataset...")
    df = load_product_data(with_images=False)

    # 1. Baseline: Handcrafted Network Features + Review Features
    X_hand = df[REVIEW_FEATURES + NETWORK_FEATURES]
    y = df[TARGET]

    X_tr_h, X_te_h, y_tr, y_te = train_test_split(
        X_hand, y, test_size=0.20, random_state=42, stratify=y
    )

    clf_base = xgb.XGBClassifier(eval_metric="logloss", random_state=42)
    clf_base.fit(X_tr_h, y_tr)
    pred_h = clf_base.predict(X_te_h)
    prob_h = clf_base.predict_proba(X_te_h)[:, 1]

    f1_base = f1_score(y_te, pred_h, pos_label=1)
    auc_base = roc_auc_score(y_te, prob_h)
    acc_base = accuracy_score(y_te, pred_h)

    # 2. Build synthetic product-product network from existing network features
    # to evaluate GraphSAGE neighborhood aggregation on products
    logger.info("Building product similarity projection graph for GraphSAGE evaluation...")
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=8, metric="cosine")
    nn.fit(df[NETWORK_FEATURES])
    adj_mat = nn.kneighbors_graph(df[NETWORK_FEATURES], mode="distance")

    G = nx.from_scipy_sparse_array(adj_mat)
    # Remap node indices to product_IDs
    mapping = {idx: df["product_ID"].iloc[idx] for idx in range(len(df))}
    G = nx.relabel_nodes(G, mapping)

    # Compute GraphSAGE Aggregations
    logger.info("Computing GraphSAGE neighborhood feature aggregations...")
    df_sage = compute_graphsage_aggregations(G, df, REVIEW_FEATURES, id_col="product_ID")

    # Combine Review Features + GraphSAGE Aggregations
    df_combined = df.merge(df_sage.drop(columns=["product_ID"]), left_index=True, right_index=True)
    sage_cols = [c for c in df_sage.columns if c != "product_ID"]
    X_sage = df_combined[REVIEW_FEATURES + sage_cols]

    X_tr_s, X_te_s, _, _ = train_test_split(
        X_sage, y, test_size=0.20, random_state=42, stratify=y
    )

    clf_sage = xgb.XGBClassifier(eval_metric="logloss", random_state=42, n_estimators=150, max_depth=5)
    clf_sage.fit(X_tr_s, y_tr)
    pred_s = clf_sage.predict(X_te_s)
    prob_s = clf_sage.predict_proba(X_te_s)[:, 1]

    f1_sage = f1_score(y_te, pred_s, pos_label=1)
    auc_sage = roc_auc_score(y_te, prob_s)
    acc_sage = accuracy_score(y_te, pred_s)

    # 3. Graph Embeddings (Spectral / Node2Vec)
    logger.info("Computing closed-form spectral graph embeddings (d=16)...")
    df_emb = compute_spectral_graph_embeddings(G, dim=16, nodes=list(df["product_ID"]))
    emb_cols = [c for c in df_emb.columns if c != "product_ID"]
    X_emb = df.merge(df_emb, on="product_ID")[REVIEW_FEATURES + emb_cols]

    X_tr_e, X_te_e, _, _ = train_test_split(
        X_emb, y, test_size=0.20, random_state=42, stratify=y
    )

    clf_emb = xgb.XGBClassifier(eval_metric="logloss", random_state=42)
    clf_emb.fit(X_tr_e, y_tr)
    pred_e = clf_emb.predict(X_te_e)
    prob_e = clf_emb.predict_proba(X_te_e)[:, 1]

    f1_emb = f1_score(y_te, pred_e, pos_label=1)
    auc_emb = roc_auc_score(y_te, prob_e)
    acc_emb = accuracy_score(y_te, pred_e)

    # Print Comparison Table
    print("\n" + "=" * 70)
    print("GRAPH ML BENCHMARK RESULTS vs PPT CLAIM (0.822 F1):")
    print("=" * 70)
    print(f"{'Method':35s} | {'Fake F1':8s} | {'ROC-AUC':8s} | {'Accuracy':8s}")
    print("-" * 70)
    print(f"{'Handcrafted Network Features + XGB':35s} | {f1_base:.4f}   | {auc_base:.4f}   | {acc_base:.4f}")
    print(f"{'Graph Embeddings (Spectral) + XGB':35s} | {f1_emb:.4f}   | {auc_emb:.4f}   | {acc_emb:.4f}")
    print(f"{'GraphSAGE Aggregations + XGB':35s} | {f1_sage:.4f}   | {auc_sage:.4f}   | {acc_sage:.4f}")
    print(f"{'PPT Stated Claim (GraphSAGE)':35s} | 0.8220   | N/A      | N/A")
    print("=" * 70)

    # Save to CSV
    res_df = pd.DataFrame([
        {"method": "Handcrafted Network Features + XGB", "fake_f1": f1_base, "roc_auc": auc_base, "accuracy": acc_base},
        {"method": "Graph Embeddings + XGB", "fake_f1": f1_emb, "roc_auc": auc_emb, "accuracy": acc_emb},
        {"method": "GraphSAGE Aggregations + XGB", "fake_f1": f1_sage, "roc_auc": auc_sage, "accuracy": acc_sage},
        {"method": "PPT Claim (Unverified)", "fake_f1": 0.8220, "roc_auc": np.nan, "accuracy": np.nan},
    ])
    out_csv = EXPERIMENT_DIR / "GRAPH_ML_BENCHMARK.csv"
    res_df.to_csv(out_csv, index=False)
    logger.info("Saved graph ML benchmark to %s", out_csv)


if __name__ == "__main__":
    main()

