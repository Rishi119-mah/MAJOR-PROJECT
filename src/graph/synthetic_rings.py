"""
Synthetic Fake-Review-Ring Simulation and Robustness Evaluation.

Injects synthetic coordinated reviewer rings into product graphs and evaluates
model detection recall and false positive rates under controlled noise.
"""

from __future__ import annotations

import logging
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, precision_score, f1_score

from src.config import RANDOM_SEED

logger = logging.getLogger(__name__)


def generate_synthetic_review_ring(
    n_ring_reviewers: int = 15,
    n_target_products: int = 5,
    connection_prob: float = 0.85,
    n_honest_reviewers: int = 100,
    n_honest_products: int = 50,
    honest_reviews_per_user: int = 3,
    random_seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, set[str], set[str]]:
    """Generate a synthetic bipartite review dataset with an injected fraudulent review ring.

    Parameters
    ----------
    n_ring_reviewers : int
        Number of coordinated fraudulent reviewer accounts.
    n_target_products : int
        Number of colluding products buying fake reviews from the ring.
    connection_prob : float
        Probability that a ring reviewer reviews a target product.
    n_honest_reviewers : int
        Number of background honest reviewers.
    n_honest_products : int
        Number of background honest products.
    honest_reviews_per_user : int
        Average number of products reviewed by an honest reviewer.

    Returns
    -------
    df_reviews : pd.DataFrame
        Table with 'reviewer_ID', 'product_ID', 'is_fake_review'.
    ring_reviewers : set[str]
    target_products : set[str]
    """
    rng = np.random.default_rng(random_seed)
    records = []

    # 1. Background honest interactions
    honest_rev_ids = [f"honest_rev_{i}" for i in range(n_honest_reviewers)]
    honest_prod_ids = [f"honest_prod_{j}" for j in range(n_honest_products)]

    for rev in honest_rev_ids:
        # Sample products with replacement or choice
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
    logger.info(
        "Generated synthetic network: %d total reviews, %d fraud reviews (%.2f%%)",
        len(df_reviews),
        df_reviews["is_fake_review"].sum(),
        df_reviews["is_fake_review"].mean() * 100,
    )
    return df_reviews, set(ring_rev_ids), set(target_prod_ids)


def evaluate_ring_robustness(
    noise_levels: list[float] = [0.0, 0.1, 0.2, 0.3, 0.5],
    ring_sizes: list[int] = [5, 10, 20, 30],
) -> pd.DataFrame:
    """Evaluate detection robustness under varying ring sizes and background noise levels."""
    from src.graph.graph_builder import build_projected_product_graph, extract_graph_features

    results = []
    for r_size in ring_sizes:
        for noise in noise_levels:
            # Noise added via additional random connections
            n_honest = int(100 * (1.0 + noise))
            df_revs, ring_revs, target_prods = generate_synthetic_review_ring(
                n_ring_reviewers=r_size,
                n_target_products=8,
                n_honest_reviewers=n_honest,
            )

            P = build_projected_product_graph(df_revs)
            feats = extract_graph_features(P, include_advanced=False)

            # Ground truth: target_prod_* is fake (1), honest_prod_* is genuine (0)
            feats["is_target"] = feats["product_ID"].apply(lambda p: 1 if "target" in str(p) else 0)

            # Detection rule based on weighted degree and clustering coefficient percentile
            deg_thresh = feats["w_degree"].quantile(0.80)
            cc_thresh = feats["clustering_coef"].quantile(0.70)
            preds = ((feats["w_degree"] >= deg_thresh) | (feats["clustering_coef"] >= cc_thresh)).astype(int)

            rec = recall_score(feats["is_target"], preds, zero_division=0)
            prec = precision_score(feats["is_target"], preds, zero_division=0)
            f1 = f1_score(feats["is_target"], preds, zero_division=0)

            results.append({
                "ring_size": r_size,
                "noise_level": noise,
                "recall": float(rec),
                "precision": float(prec),
                "f1_score": float(f1),
                "mean_target_w_degree": float(feats[feats["is_target"] == 1]["w_degree"].mean()),
                "mean_honest_w_degree": float(feats[feats["is_target"] == 0]["w_degree"].mean()),
            })

    df_res = pd.DataFrame(results)
    logger.info("Synthetic robustness evaluation completed across %d scenarios.", len(df_res))
    return df_res

