"""
Synthetic Fake-Review-Ring Simulation and Robustness Evaluation.

Injects synthetic coordinated reviewer rings into product graphs and evaluates
model detection recall, precision, specificity, and false positive rates under controlled noise.
"""

from __future__ import annotations

import logging
from typing import Any, Tuple, Set, List
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, confusion_matrix

from src.config import RANDOM_SEED

logger = logging.getLogger(__name__)


def generate_synthetic_review_ring(
    n_ring_reviewers: int = 15,
    n_target_products: int = 8,
    connection_prob: float = 0.85,
    n_honest_reviewers: int = 100,
    n_honest_products: int = 50,
    honest_reviews_per_user: int = 3,
    random_seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, Set[str], Set[str]]:
    """Generate a synthetic bipartite review dataset with an injected fraudulent review ring."""
    rng = np.random.default_rng(random_seed)
    records = []

    # 1. Background honest interactions
    honest_rev_ids = [f"honest_rev_{i}" for i in range(n_honest_reviewers)]
    honest_prod_ids = [f"honest_prod_{j}" for j in range(n_honest_products)]

    for rev in honest_rev_ids:
        num_reviews = rng.poisson(honest_reviews_per_user) + 1
        prods = rng.choice(honest_prod_ids, size=min(num_reviews, len(honest_prod_ids)), replace=False)
        for p in prods:
            records.append({
                "reviewer_ID": rev,
                "product_ID": p,
                "is_fake_review": 0,
            })

    # 2. Injected coordinated fraud ring
    ring_rev_ids = [f"ring_rev_{i}" for i in range(n_ring_reviewers)]
    target_prod_ids = [f"target_prod_{j}" for j in range(n_target_products)]

    for r in ring_rev_ids:
        for p in target_prod_ids:
            if rng.random() < connection_prob:
                records.append({
                    "reviewer_ID": r,
                    "product_ID": p,
                    "is_fake_review": 1,
                })

    df_reviews = pd.DataFrame(records)
    logger.debug(
        "Generated synthetic network: %d total reviews, %d fraud reviews (%.2f%%)",
        len(df_reviews),
        df_reviews["is_fake_review"].sum(),
        df_reviews["is_fake_review"].mean() * 100,
    )
    return df_reviews, set(ring_rev_ids), set(target_prod_ids)


def evaluate_ring_robustness(
    noise_levels: List[float] = [0.0, 0.1, 0.2, 0.3, 0.5],
    ring_sizes: List[int] = [5, 10, 20, 30, 50],
    random_seeds: List[int] = [42, 123],
) -> pd.DataFrame:
    """Evaluate detection robustness under varying ring sizes and background noise levels.

    Evaluates both Recall AND Precision, Specificity, FPR, and Confusion Matrix metrics.
    """
    from src.graph.graph_builder import build_projected_product_graph, extract_graph_features

    results = []
    for r_size in ring_sizes:
        for noise in noise_levels:
            for seed in random_seeds:
                n_honest = int(100 * (1.0 + noise))
                df_revs, ring_revs, target_prods = generate_synthetic_review_ring(
                    n_ring_reviewers=r_size,
                    n_target_products=8,
                    n_honest_reviewers=n_honest,
                    random_seed=seed,
                )

                P = build_projected_product_graph(df_revs)
                feats = extract_graph_features(P, include_advanced=False)

                # Ground truth: target_prod_* is fake (1), honest_prod_* is genuine (0)
                y_true = feats["product_ID"].apply(lambda p: 1 if "target" in str(p) else 0).values

                # Heuristic 1: High-Recall Threshold (Degree >= 80th pct OR CC >= 70th pct)
                deg_thresh = feats["w_degree"].quantile(0.80)
                cc_thresh = feats["clustering_coef"].quantile(0.70)
                preds_high_recall = ((feats["w_degree"] >= deg_thresh) | (feats["clustering_coef"] >= cc_thresh)).astype(int).values

                # Heuristic 2: Balanced Threshold (Degree >= 85th pct AND CC >= 75th pct)
                deg_b_thresh = feats["w_degree"].quantile(0.85)
                cc_b_thresh = feats["clustering_coef"].quantile(0.75)
                preds_balanced = ((feats["w_degree"] >= deg_b_thresh) & (feats["clustering_coef"] >= cc_b_thresh)).astype(int).values

                for h_name, preds in [("High-Recall Percentile Heuristic", preds_high_recall), ("Balanced Dual-Threshold Heuristic", preds_balanced)]:
                    cm = confusion_matrix(y_true, preds, labels=[0, 1])
                    tn, fp, fn, tp = cm.ravel()

                    rec = recall_score(y_true, preds, pos_label=1, zero_division=0)
                    prec = precision_score(y_true, preds, pos_label=1, zero_division=0)
                    f1 = f1_score(y_true, preds, pos_label=1, zero_division=0)
                    acc = accuracy_score(y_true, preds)
                    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
                    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

                    results.append({
                        "heuristic_strategy": h_name,
                        "ring_size": r_size,
                        "noise_level": noise,
                        "random_seed": seed,
                        "recall": float(rec),
                        "precision": float(prec),
                        "f1_score": float(f1),
                        "specificity": float(spec),
                        "false_positive_rate": float(fpr),
                        "accuracy": float(acc),
                        "tp": int(tp),
                        "fp": int(fp),
                        "tn": int(tn),
                        "fn": int(fn),
                        "mean_target_w_degree": float(feats[y_true == 1]["w_degree"].mean()) if np.any(y_true == 1) else 0.0,
                        "mean_honest_w_degree": float(feats[y_true == 0]["w_degree"].mean()) if np.any(y_true == 0) else 0.0,
                    })

    df_res = pd.DataFrame(results)
    logger.info("Synthetic robustness evaluation completed across %d test scenarios.", len(df_res))
    return df_res
