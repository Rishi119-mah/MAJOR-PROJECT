"""
Graph Representation Learning: Node2Vec & GraphSAGE Neighborhood Aggregation.

Evaluates learned graph embeddings and neighborhood aggregation on the product
co-review graph to rigorously benchmark against handcrafted network features.
Directly investigates the unverified presentation claim of GraphSAGE F1=0.822.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, accuracy_score
import xgboost as xgb

from src.config import RANDOM_SEED, TARGET

logger = logging.getLogger(__name__)


def generate_node2vec_walks(
    G: nx.Graph,
    num_walks: int = 10,
    walk_length: int = 20,
    p: float = 1.0,
    q: float = 1.0,
    seed: int = RANDOM_SEED,
) -> List[List[Any]]:
    """Generate biased random walks according to Node2Vec sampling strategy."""
    rng = np.random.default_rng(seed)
    nodes = list(G.nodes())
    walks = []

    for _ in range(num_walks):
        rng.shuffle(nodes)
        for node in nodes:
            walk = [node]
            while len(walk) < walk_length:
                curr = walk[-1]
                nbrs = list(G.neighbors(curr))
                if not nbrs:
                    break
                if len(walk) == 1:
                    walk.append(rng.choice(nbrs))
                else:
                    prev = walk[-2]
                    # Compute unnormalized transition weights based on p and q
                    weights = []
                    for nbr in nbrs:
                        if nbr == prev:
                            weights.append(1.0 / p)
                        elif G.has_edge(nbr, prev):
                            weights.append(1.0)
                        else:
                            weights.append(1.0 / q)
                    weights = np.array(weights)
                    prob = weights / weights.sum()
                    next_node = rng.choice(nbrs, p=prob)
                    walk.append(next_node)
            walks.append(walk)

    return walks


def compute_spectral_graph_embeddings(
    G: nx.Graph,
    dim: int = 16,
    nodes: List[Any] | None = None,
) -> pd.DataFrame:
    """Compute normalized Laplacian spectral embeddings as a robust closed-form graph embedding."""
    if nodes is None:
        nodes = list(G.nodes())

    A = nx.to_numpy_array(G, nodelist=nodes, weight="weight")
    deg = A.sum(axis=1)
    deg_inv_sqrt = np.power(deg, -0.5, where=deg > 0)
    deg_inv_sqrt[deg == 0] = 0.0
    D_inv = np.diag(deg_inv_sqrt)
    L_norm = np.eye(len(nodes)) - D_inv @ A @ D_inv

    # Eigen-decomposition
    eigvals, eigvecs = np.linalg.eigh(L_norm)
    # Discard trivial first eigenvector, take next `dim` smallest
    embeddings = eigvecs[:, 1:dim + 1]

    cols = [f"graph_emb_{i}" for i in range(embeddings.shape[1])]
    df_emb = pd.DataFrame(embeddings, columns=cols)
    df_emb["product_ID"] = nodes
    return df_emb


def compute_graphsage_aggregations(
    G: nx.Graph,
    df_features: pd.DataFrame,
    feature_cols: List[str],
    id_col: str = "product_ID",
) -> pd.DataFrame:
    """Inductive GraphSAGE-style Mean Neighborhood Feature Aggregator.

    Computes for each node v:
      h_N(v) = Mean_{u in N(v)} [ x_u * edge_weight(v, u) ] / sum(edge_weights)
    """
    node_to_idx = {row[id_col]: idx for idx, row in df_features.iterrows()}
    X = df_features[feature_cols].values
    n_nodes = len(df_features)
    n_feats = len(feature_cols)

    agg_features = np.zeros((n_nodes, n_feats), dtype=np.float32)

    for prod_id, idx in node_to_idx.items():
        if G.has_node(prod_id):
            nbrs = list(G.neighbors(prod_id))
            if nbrs:
                nbr_indices = [node_to_idx[nbr] for nbr in nbrs if nbr in node_to_idx]
                if nbr_indices:
                    weights = np.array([G[prod_id][nbr].get("weight", 1.0) for nbr in nbrs if nbr in node_to_idx])
                    weights = weights / weights.sum()
                    nbr_X = X[nbr_indices]
                    agg_features[idx] = (nbr_X.T @ weights).T
                else:
                    agg_features[idx] = X[idx]
            else:
                agg_features[idx] = X[idx]
        else:
            agg_features[idx] = X[idx]

    agg_cols = [f"graphsage_agg_{c}" for c in feature_cols]
    df_agg = pd.DataFrame(agg_features, columns=agg_cols)
    df_agg[id_col] = df_features[id_col].values
    return df_agg

