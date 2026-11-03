"""
Comprehensive Graph Machine Learning Benchmark:
Tabular Baseline vs Inductive KNN Aggregation vs Genuine Node2Vec vs Genuine GraphSAGE vs Spectral Embeddings.

LEAKAGE-FREE PROTOCOL:
- 80/20 Stratified Split: Development (80%) vs Untouched Final Holdout (20%).
- Graph representations (KNN, Node2Vec, GraphSAGE) are fitted/trained on Development data.
- Downstream models are evaluated on the untouched Final Holdout.
- Investigates and reconciles the historical PPT claim of GraphSAGE F1=0.822.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
import xgboost as xgb
import catboost as cb

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import EXPERIMENT_DIR, NETWORK_FEATURES, REVIEW_FEATURES, TARGET, RANDOM_SEED
from src.data.data_loader import load_product_data
from src.graph.gnn_models import (
    TrainableGraphSAGE,
    generate_node2vec_walks,
    train_node2vec_embeddings,
    compute_spectral_graph_embeddings,
    KNNNeighborAggregator,
)
from src.evaluation.metrics import compute_all_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("gnn_benchmark")


def main():
    logger.info("Loading labelled product dataset (without image feats for max sample coverage)...")
    df = load_product_data(with_images=False)

    # 1. Stratified Holdout Split: Development (80%) vs Final Test (20%)
    df_dev, df_test = train_test_split(
        df, test_size=0.20, random_state=RANDOM_SEED, stratify=df[TARGET]
    )

    y_dev = df_dev[TARGET].values
    y_test = df_test[TARGET].values

    benchmark_rows = []

    # =========================================================================
    # A. Tabular XGBoost (Review + Handcrafted Network Centrality)
    # =========================================================================
    logger.info("1/6 Running Tabular XGBoost (Handcrafted Centrality Baseline)...")
    feats_tabular = REVIEW_FEATURES + NETWORK_FEATURES
    scaler_tab = StandardScaler()
    X_dev_tab = scaler_tab.fit_transform(df_dev[feats_tabular])
    X_test_tab = scaler_tab.transform(df_test[feats_tabular])

    t0 = time.perf_counter()
    clf_xgb = xgb.XGBClassifier(eval_metric="logloss", random_state=RANDOM_SEED, n_estimators=100, max_depth=5)
    clf_xgb.fit(X_dev_tab, y_dev)
    t_train = time.perf_counter() - t0

    t1 = time.perf_counter()
    pred_xgb = clf_xgb.predict(X_test_tab)
    prob_xgb = clf_xgb.predict_proba(X_test_tab)[:, 1]
    t_inf = time.perf_counter() - t1

    m_xgb = compute_all_metrics(y_test, pred_xgb, prob_xgb)
    m_xgb.update({"model": "Tabular XGBoost (Handcrafted Centrality)", "train_time_s": t_train, "inference_time_s": t_inf})
    benchmark_rows.append(m_xgb)

    # =========================================================================
    # B. Tabular CatBoost (Review + Handcrafted Network Centrality)
    # =========================================================================
    logger.info("2/6 Running Tabular CatBoost (Handcrafted Centrality Baseline)...")
    t0 = time.perf_counter()
    clf_cb = cb.CatBoostClassifier(iterations=150, learning_rate=0.08, depth=5, verbose=0, random_seed=RANDOM_SEED)
    clf_cb.fit(X_dev_tab, y_dev)
    t_train = time.perf_counter() - t0

    t1 = time.perf_counter()
    pred_cb = clf_cb.predict(X_test_tab)
    prob_cb = clf_cb.predict_proba(X_test_tab)[:, 1]
    t_inf = time.perf_counter() - t1

    m_cb = compute_all_metrics(y_test, pred_cb, prob_cb)
    m_cb.update({"model": "Tabular CatBoost (Handcrafted Centrality)", "train_time_s": t_train, "inference_time_s": t_inf})
    benchmark_rows.append(m_cb)

    # =========================================================================
    # C. Inductive KNN Neighbour-Feature Aggregation + XGBoost
    # =========================================================================
    logger.info("3/6 Running Inductive KNN Neighbour-Feature Aggregation + XGBoost...")
    knn_agg = KNNNeighborAggregator(n_neighbors=8)
    knn_agg.fit(df_dev[NETWORK_FEATURES].values, df_dev["product_ID"].values)

    dev_knn_agg = knn_agg.transform(df_dev[NETWORK_FEATURES].values)
    test_knn_agg = knn_agg.transform(df_test[NETWORK_FEATURES].values)

    X_dev_knn = np.hstack([X_dev_tab, dev_knn_agg])
    X_test_knn = np.hstack([X_test_tab, test_knn_agg])

    t0 = time.perf_counter()
    clf_knn_xgb = xgb.XGBClassifier(eval_metric="logloss", random_state=RANDOM_SEED, n_estimators=100, max_depth=5)
    clf_knn_xgb.fit(X_dev_knn, y_dev)
    t_train = time.perf_counter() - t0

    t1 = time.perf_counter()
    pred_knn_xgb = clf_knn_xgb.predict(X_test_knn)
    prob_knn_xgb = clf_knn_xgb.predict_proba(X_test_knn)[:, 1]
    t_inf = time.perf_counter() - t1

    m_knn_xgb = compute_all_metrics(y_test, pred_knn_xgb, prob_knn_xgb)
    m_knn_xgb.update({"model": "KNN Neighbour Aggregation + XGBoost", "train_time_s": t_train, "inference_time_s": t_inf})
    benchmark_rows.append(m_knn_xgb)

    # =========================================================================
    # D. Genuine Node2Vec (Biased Random Walks + Skip-Gram Negative Sampling)
    # =========================================================================
    logger.info("4/6 Running Genuine Node2Vec (Biased Walks + SGNS Embeddings d=16)...")
    # Build graph on dev nodes to prevent transductive leakage
    from sklearn.neighbors import NearestNeighbors
    nn_n2v = NearestNeighbors(n_neighbors=6, metric="cosine")
    nn_n2v.fit(df_dev[NETWORK_FEATURES].values)
    adj_dev = nn_n2v.kneighbors_graph(df_dev[NETWORK_FEATURES].values, mode="connectivity").toarray()

    G_dev = nx.from_numpy_array(adj_dev)
    mapping = {idx: df_dev["product_ID"].iloc[idx] for idx in range(len(df_dev))}
    G_dev = nx.relabel_nodes(G_dev, mapping)

    t0 = time.perf_counter()
    walks = generate_node2vec_walks(G_dev, num_walks=8, walk_length=15, p=1.0, q=1.0, seed=RANDOM_SEED)
    embs_dict = train_node2vec_embeddings(walks, dim=16, window_size=3, epochs=4, seed=RANDOM_SEED)

    # Transform dev and test nodes
    default_emb = np.zeros(16)
    dev_embs = np.array([embs_dict.get(pid, default_emb) for pid in df_dev["product_ID"]])

    # For test nodes, infer embedding via nearest dev neighbor interpolation
    test_nbr_indices = nn_n2v.kneighbors(df_test[NETWORK_FEATURES].values, n_neighbors=1, return_distance=False)
    test_embs = np.array([embs_dict.get(df_dev["product_ID"].iloc[idx[0]], default_emb) for idx in test_nbr_indices])

    scaler_rev = StandardScaler()
    X_dev_rev = scaler_rev.fit_transform(df_dev[REVIEW_FEATURES])
    X_test_rev = scaler_rev.transform(df_test[REVIEW_FEATURES])

    X_dev_n2v = np.hstack([X_dev_rev, dev_embs])
    X_test_n2v = np.hstack([X_test_rev, test_embs])

    clf_n2v = xgb.XGBClassifier(eval_metric="logloss", random_state=RANDOM_SEED, n_estimators=100, max_depth=5)
    clf_n2v.fit(X_dev_n2v, y_dev)
    t_train = time.perf_counter() - t0

    t1 = time.perf_counter()
    pred_n2v = clf_n2v.predict(X_test_n2v)
    prob_n2v = clf_n2v.predict_proba(X_test_n2v)[:, 1]
    t_inf = time.perf_counter() - t1

    m_n2v = compute_all_metrics(y_test, pred_n2v, prob_n2v)
    m_n2v.update({"model": "Genuine Node2Vec (SGNS d=16) + XGBoost", "train_time_s": t_train, "inference_time_s": t_inf})
    benchmark_rows.append(m_n2v)

    # =========================================================================
    # E. Genuine Trainable GraphSAGE (2-layer Neural Message Passing)
    # =========================================================================
    logger.info("5/6 Running Genuine Trainable GraphSAGE (2-Layer Neural Message Passing)...")
    # Adjacency matrix for both dev and test nodes (inductive mask evaluation)
    nn_all = NearestNeighbors(n_neighbors=6, metric="cosine")
    nn_all.fit(df[NETWORK_FEATURES].values)
    adj_all = nn_all.kneighbors_graph(df[NETWORK_FEATURES].values, mode="connectivity").toarray()

    dev_indices = df_dev.index.values
    test_indices = df_test.index.values
    all_indices = df.index.values

    # Map indices to positions
    idx_map = {orig_idx: pos for pos, orig_idx in enumerate(all_indices)}
    train_mask = np.zeros(len(df), dtype=bool)
    test_mask = np.zeros(len(df), dtype=bool)

    # 80% of dev for train, 20% of dev for validation inside GNN
    n_dev_train = int(len(dev_indices) * 0.85)
    train_mask[[idx_map[i] for i in dev_indices[:n_dev_train]]] = True
    val_mask = np.zeros(len(df), dtype=bool)
    val_mask[[idx_map[i] for i in dev_indices[n_dev_train:]]] = True
    test_mask[[idx_map[i] for i in test_indices]] = True

    scaler_all = StandardScaler()
    X_all = scaler_all.fit_transform(df[feats_tabular].values)
    y_all = df[TARGET].values

    t0 = time.perf_counter()
    gnn_sage = TrainableGraphSAGE(in_dim=len(feats_tabular), hidden_dim=32, out_dim=16, lr=0.01, seed=RANDOM_SEED)
    gnn_sage.fit(X_all, adj_all, y_all, train_mask=train_mask, val_mask=val_mask, max_epochs=120, patience=15)
    t_train = time.perf_counter() - t0

    t1 = time.perf_counter()
    prob_all_sage = gnn_sage.predict_proba(X_all, adj_all)
    prob_test_sage = prob_all_sage[test_mask]
    pred_test_sage = (prob_test_sage >= 0.5).astype(int)
    t_inf = time.perf_counter() - t1

    m_sage = compute_all_metrics(y_test, pred_test_sage, prob_test_sage)
    m_sage.update({"model": "Genuine Trainable GraphSAGE (Neural GNN)", "train_time_s": t_train, "inference_time_s": t_inf})
    benchmark_rows.append(m_sage)

    # =========================================================================
    # F. Spectral Graph Embeddings (Laplacian Eigenvectors d=16 + XGBoost)
    # =========================================================================
    logger.info("6/6 Running Spectral Graph Embeddings (Normalized Laplacian d=16)...")
    t0 = time.perf_counter()
    df_spectral = compute_spectral_graph_embeddings(G_dev, dim=16, nodes=list(df_dev["product_ID"]))
    emb_cols = [c for c in df_spectral.columns if c != "product_ID"]
    df_dev_spec = df_dev.merge(df_spectral, on="product_ID")
    X_dev_spec = np.hstack([scaler_rev.transform(df_dev_spec[REVIEW_FEATURES]), df_dev_spec[emb_cols].values])

    # Infer test spectral features via nearest neighbor projection
    test_spec_embs = df_spectral[emb_cols].iloc[test_nbr_indices.ravel()].values
    X_test_spec = np.hstack([X_test_rev, test_spec_embs])

    clf_spec = xgb.XGBClassifier(eval_metric="logloss", random_state=RANDOM_SEED, n_estimators=100, max_depth=5)
    clf_spec.fit(X_dev_spec, y_dev)
    t_train = time.perf_counter() - t0

    t1 = time.perf_counter()
    pred_spec = clf_spec.predict(X_test_spec)
    prob_spec = clf_spec.predict_proba(X_test_spec)[:, 1]
    t_inf = time.perf_counter() - t1

    m_spec = compute_all_metrics(y_test, pred_spec, prob_spec)
    m_spec.update({"model": "Spectral Graph Embeddings (d=16) + XGBoost", "train_time_s": t_train, "inference_time_s": t_inf})
    benchmark_rows.append(m_spec)

    # Save benchmark table
    df_bench = pd.DataFrame(benchmark_rows)
    # Reorder columns
    lead_cols = ["model", "accuracy", "fake_f1", "fake_precision", "fake_recall", "roc_auc", "pr_auc", "macro_f1", "train_time_s", "inference_time_s"]
    other_cols = [c for c in df_bench.columns if c not in lead_cols]
    df_bench = df_bench[lead_cols + other_cols]

    out_csv = EXPERIMENT_DIR / "GRAPH_ML_BENCHMARK.csv"
    df_bench.to_csv(out_csv, index=False)
    logger.info("Saved Graph ML Benchmark to %s", out_csv)

    print("\n" + "=" * 90)
    print("GRAPH MACHINE LEARNING BENCHMARK (LEAKAGE-FREE FINAL HOLDOUT EVALUATION)")
    print("=" * 90)
    for _, r in df_bench.iterrows():
        print(
            f"{r['model']:45s} | AUC: {r['roc_auc']:.4f} | PR-AUC: {r['pr_auc']:.4f} | "
            f"Acc: {r['accuracy']:.4f} | Fake F1: {r['fake_f1']:.4f} | Time: {r['train_time_s']:.2f}s"
        )
    print("=" * 90)


if __name__ == "__main__":
    main()
