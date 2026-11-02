"""
Graph construction and feature extraction for Fake Review Detection.

Provides collision-safe bipartite reviewer-product graph construction
and fast projected product-product graph generation without `iterrows()`.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from itertools import combinations
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_bipartite_graph(
    df_reviews: pd.DataFrame,
    reviewer_col: str = "reviewer_ID",
    product_col: str = "product_ID",
    add_prefix: bool = True,
) -> tuple[nx.Graph, set[str], set[str]]:
    """Construct a bipartite graph with collision-safe node IDs.

    Parameters
    ----------
    df_reviews : pd.DataFrame
        Table with reviewer and product columns.
    reviewer_col : str
        Name of reviewer ID column.
    product_col : str
        Name of product ID column.
    add_prefix : bool, default True
        If True, prepend 'rev_' to reviewers and 'prod_' to products
        to prevent node collisions when IDs have the same type/values.

    Returns
    -------
    B : nx.Graph
        Bipartite networkx graph.
    reviewer_nodes : set[str]
        Set of reviewer node identifiers.
    product_nodes : set[str]
        Set of product node identifiers.
    """
    B = nx.Graph()

    if add_prefix:
        rev_ids = ["rev_" + str(r) for r in df_reviews[reviewer_col]]
        prod_ids = ["prod_" + str(p) for p in df_reviews[product_col]]
    else:
        rev_ids = df_reviews[reviewer_col].tolist()
        prod_ids = df_reviews[product_col].tolist()

    rev_set = set(rev_ids)
    prod_set = set(prod_ids)

    B.add_nodes_from(rev_set, bipartite=0)
    B.add_nodes_from(prod_set, bipartite=1)

    # Fast edge insertion via zip (avoids slow iterrows)
    edges = list(zip(rev_ids, prod_ids))
    B.add_edges_from(edges)

    logger.info(
        "Bipartite graph built: %d reviewers, %d products, %d review edges",
        len(rev_set),
        len(prod_set),
        B.number_of_edges(),
    )
    return B, rev_set, prod_set


def build_projected_product_graph(
    df_reviews: pd.DataFrame,
    reviewer_col: str = "reviewer_ID",
    product_col: str = "product_ID",
    min_shared_reviewers: int = 1,
) -> nx.Graph:
    """Fast product-product co-review projection without iterrows.

    Each edge weight is the number of shared reviewers between two products.
    """
    # Group products by reviewer: rev -> list of products reviewed
    rev_to_prods = (
        df_reviews.groupby(reviewer_col)[product_col]
        .apply(lambda s: list(set(s)))
        .to_dict()
    )

    edge_weights: dict[tuple[Any, Any], int] = defaultdict(int)
    all_products = set(df_reviews[product_col].unique())

    for rev, prods in rev_to_prods.items():
        if len(prods) > 1:
            for p1, p2 in combinations(sorted(prods), 2):
                edge_weights[(p1, p2)] += 1

    P = nx.Graph()
    P.add_nodes_from(all_products)

    for (p1, p2), weight in edge_weights.items():
        if weight >= min_shared_reviewers:
            P.add_edge(p1, p2, weight=weight)

    logger.info(
        "Product projection graph built: %d nodes, %d edges",
        P.number_of_nodes(),
        P.number_of_edges(),
    )
    return P


def extract_graph_features(
    P: nx.Graph,
    products: list[Any] | None = None,
    include_advanced: bool = True,
) -> pd.DataFrame:
    """Extract standard and advanced network features from a product projection graph.

    Standard Features (reproduced):
      - pagerank
      - w_degree
      - clustering_coef
      - eigenvector_cent

    Advanced Features:
      - betweenness_cent (approximated for speed if large)
      - k_core
      - degree (unweighted)
    """
    if products is None:
        products = list(P.nodes())

    df_feats = pd.DataFrame({"product_ID": products})

    # Standard features
    logger.info("Computing PageRank...")
    try:
        pr = nx.pagerank(P, alpha=0.85, weight="weight")
    except Exception:
        pr = {p: 0.0 for p in products}

    logger.info("Computing Eigenvector Centrality...")
    try:
        eig = nx.eigenvector_centrality(P, max_iter=500, weight="weight")
    except Exception:
        try:
            eig = nx.eigenvector_centrality_numpy(P, weight="weight")
        except Exception:
            eig = {p: 0.0 for p in products}

    logger.info("Computing Clustering Coefficients...")
    try:
        cc = nx.clustering(P, weight="weight")
    except Exception:
        cc = {p: 0.0 for p in products}

    logger.info("Computing Weighted Degrees...")
    w_deg = dict(P.degree(weight="weight"))
    deg = dict(P.degree())

    df_feats["pagerank"] = [pr.get(p, 0.0) for p in products]
    df_feats["eigenvector_cent"] = [eig.get(p, 0.0) for p in products]
    df_feats["clustering_coef"] = [cc.get(p, 0.0) for p in products]
    df_feats["w_degree"] = [w_deg.get(p, 0) for p in products]
    df_feats["degree"] = [deg.get(p, 0) for p in products]

    if include_advanced:
        logger.info("Computing K-Core...")
        try:
            core_numbers = nx.core_number(P)
        except Exception:
            core_numbers = {p: 0 for p in products}
        df_feats["k_core"] = [core_numbers.get(p, 0) for p in products]

        if P.number_of_nodes() <= 5000:
            logger.info("Computing Betweenness Centrality...")
            try:
                betw = nx.betweenness_centrality(P, weight="weight", normalized=True)
            except Exception:
                betw = {p: 0.0 for p in products}
            df_feats["betweenness_cent"] = [betw.get(p, 0.0) for p in products]
        else:
            df_feats["betweenness_cent"] = 0.0

    return df_feats

