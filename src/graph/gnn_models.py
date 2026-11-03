"""
Graph Machine Learning: Genuine GraphSAGE, Node2Vec (SGNS), Spectral Embeddings,
and Inductive KNN Neighbor Feature Aggregation.

Provides mathematically rigorous, trainable graph representation learning models:
1. TrainableGraphSAGE: 2-layer inductive neural message-passing network with Adam optimizer,
   backpropagation, loss tracking, and early stopping.
2. Node2Vec: Second-order biased random walks (p, q) + Skip-Gram with Negative Sampling (SGNS).
3. SpectralGraphEmbeddings: Normalized Laplacian eigen-decomposition.
4. KNNNeighborAggregator: Inductive KNN neighbor-feature aggregation (renamed from legacy GraphSAGE).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score

from src.config import RANDOM_SEED, TARGET

logger = logging.getLogger(__name__)


# =====================================================================
# 1. GENUINE TRAINABLE GRAPHSAGE (Neural Message Passing)
# =====================================================================

class TrainableGraphSAGE:
    """Two-layer Inductive GraphSAGE Neural Network with Mean Aggregation and Adam Optimizer.

    Layer 1: h_v^(1) = ReLU( W_1 * CONCAT(x_v, MEAN_{u in N(v)}(x_u)) + b_1 )
    Layer 2: h_v^(2) = ReLU( W_2 * CONCAT(h_v^(1), MEAN_{u in N(v)}(h_u^(1))) + b_2 )
    Readout: y_hat = Sigmoid( W_cls * h_v^(2) + b_cls )
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 32,
        out_dim: int = 16,
        lr: float = 0.01,
        weight_decay: float = 1e-4,
        seed: int = RANDOM_SEED,
    ):
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.seed = seed

        rng = np.random.default_rng(seed)

        # Xavier/He parameter initialization
        scale1 = np.sqrt(2.0 / (2 * in_dim))
        self.W1 = rng.normal(0, scale1, (2 * in_dim, hidden_dim)).astype(np.float64)
        self.b1 = np.zeros(hidden_dim, dtype=np.float64)

        scale2 = np.sqrt(2.0 / (2 * hidden_dim))
        self.W2 = rng.normal(0, scale2, (2 * hidden_dim, out_dim)).astype(np.float64)
        self.b2 = np.zeros(out_dim, dtype=np.float64)

        scale_cls = np.sqrt(2.0 / out_dim)
        self.W_cls = rng.normal(0, scale_cls, (out_dim, 1)).astype(np.float64)
        self.b_cls = np.zeros(1, dtype=np.float64)

        # Adam optimizer state
        self.m = {k: np.zeros_like(v) for k, v in self._get_params().items()}
        self.v = {k: np.zeros_like(v) for k, v in self._get_params().items()}
        self.t = 0

    def _get_params(self) -> Dict[str, np.ndarray]:
        return {
            "W1": self.W1, "b1": self.b1,
            "W2": self.W2, "b2": self.b2,
            "W_cls": self.W_cls, "b_cls": self.b_cls,
        }

    @staticmethod
    def _normalize_adj(adj: np.ndarray) -> np.ndarray:
        """Add self-loops and compute row-normalized adjacency D^-1 (A + I)."""
        A_tilde = adj + np.eye(adj.shape[0])
        deg = np.sum(A_tilde, axis=1, keepdims=True)
        deg[deg == 0] = 1.0
        return A_tilde / deg

    def forward(
        self, X: np.ndarray, A_norm: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Forward pass through 2 GraphSAGE layers and classifier head."""
        # Layer 1
        neigh_X = A_norm @ X
        concat1 = np.hstack([X, neigh_X])
        z1 = concat1 @ self.W1 + self.b1
        h1 = np.maximum(0.0, z1)  # ReLU

        # Layer 2
        neigh_h1 = A_norm @ h1
        concat2 = np.hstack([h1, neigh_h1])
        z2 = concat2 @ self.W2 + self.b2
        h2 = np.maximum(0.0, z2)  # ReLU (embeddings)

        # Classifier head
        logits = h2 @ self.W_cls + self.b_cls
        p = 1.0 / (1.0 + np.exp(-np.clip(logits, -25.0, 25.0)))
        return concat1, z1, concat2, z2, p

    def fit(
        self,
        X: np.ndarray,
        adj: np.ndarray,
        y: np.ndarray,
        train_mask: np.ndarray,
        val_mask: Optional[np.ndarray] = None,
        max_epochs: int = 150,
        patience: int = 20,
    ) -> Dict[str, List[float]]:
        """Train GraphSAGE using analytical backpropagation on training nodes."""
        A_norm = self._normalize_adj(adj)
        y = y.reshape(-1, 1).astype(np.float64)

        best_val_loss = float("inf")
        best_weights = {k: v.copy() for k, v in self._get_params().items()}
        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "val_f1": []}
        patience_counter = 0

        for epoch in range(1, max_epochs + 1):
            self.t += 1
            concat1, z1, concat2, z2, p = self.forward(X, A_norm)

            # Compute loss on training mask
            eps = 1e-12
            train_p = p[train_mask]
            train_y = y[train_mask]
            n_tr = np.sum(train_mask)

            bce_loss = -np.mean(train_y * np.log(train_p + eps) + (1.0 - train_y) * np.log(1.0 - train_p + eps))
            reg_loss = 0.5 * self.weight_decay * (
                np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2) + np.sum(self.W_cls ** 2)
            )
            total_loss = bce_loss + reg_loss
            history["train_loss"].append(float(total_loss))

            # Backward pass
            # Gradient of loss w.r.t logits (only for train nodes)
            dlogits = np.zeros_like(p)
            dlogits[train_mask] = (train_p - train_y) / n_tr

            # Head gradients
            h2 = np.maximum(0.0, z2)
            dW_cls = h2.T @ dlogits + self.weight_decay * self.W_cls
            db_cls = np.sum(dlogits, axis=0)

            # Backprop into Layer 2
            dh2 = dlogits @ self.W_cls.T
            dz2 = dh2 * (z2 > 0).astype(np.float64)
            dW2 = concat2.T @ dz2 + self.weight_decay * self.W2
            db2 = np.sum(dz2, axis=0)

            # Gradient w.r.t concat2 = [h1, A_norm @ h1]
            dconcat2 = dz2 @ self.W2.T
            dh1_self = dconcat2[:, :self.hidden_dim]
            dh1_neigh = dconcat2[:, self.hidden_dim:]
            dh1 = dh1_self + A_norm.T @ dh1_neigh

            # Backprop into Layer 1
            dz1 = dh1 * (z1 > 0).astype(np.float64)
            dW1 = concat1.T @ dz1 + self.weight_decay * self.W1
            db1 = np.sum(dz1, axis=0)

            grads = {
                "W1": dW1, "b1": db1,
                "W2": dW2, "b2": db2,
                "W_cls": dW_cls, "b_cls": db_cls,
            }

            # Adam Optimizer update
            beta1, beta2, eps_adam = 0.9, 0.999, 1e-8
            for k, param in self._get_params().items():
                g = grads[k]
                self.m[k] = beta1 * self.m[k] + (1.0 - beta1) * g
                self.v[k] = beta2 * self.v[k] + (1.0 - beta2) * (g ** 2)
                m_hat = self.m[k] / (1.0 - beta1 ** self.t)
                v_hat = self.v[k] / (1.0 - beta2 ** self.t)
                param -= self.lr * m_hat / (np.sqrt(v_hat) + eps_adam)

            # Validation evaluation
            if val_mask is not None and np.any(val_mask):
                val_p = p[val_mask]
                val_y = y[val_mask]
                v_loss = -np.mean(val_y * np.log(val_p + eps) + (1.0 - val_y) * np.log(1.0 - val_p + eps))
                val_pred = (val_p >= 0.5).astype(int)
                val_f1 = f1_score(val_y, val_pred, pos_label=1, zero_division=0)
                history["val_loss"].append(float(v_loss))
                history["val_f1"].append(float(val_f1))

                if v_loss < best_val_loss:
                    best_val_loss = v_loss
                    best_weights = {k: v.copy() for k, v in self._get_params().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.debug("GraphSAGE early stopping triggered at epoch %d", epoch)
                        break

        # Restore best weights
        for k, v in best_weights.items():
            self._get_params()[k][:] = v

        return history

    def predict_proba(self, X: np.ndarray, adj: np.ndarray) -> np.ndarray:
        """Return probability of fake review buyer for each node."""
        A_norm = self._normalize_adj(adj)
        _, _, _, _, p = self.forward(X, A_norm)
        return p.ravel()

    def predict(self, X: np.ndarray, adj: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return binary class prediction."""
        return (self.predict_proba(X, adj) >= threshold).astype(int)

    def extract_embeddings(self, X: np.ndarray, adj: np.ndarray) -> np.ndarray:
        """Extract learned 16-dimensional node representations from Layer 2."""
        A_norm = self._normalize_adj(adj)
        _, _, _, z2, _ = self.forward(X, A_norm)
        return np.maximum(0.0, z2)


# =====================================================================
# 2. GENUINE NODE2VEC (Biased Walks + Skip-Gram Negative Sampling)
# =====================================================================

def generate_node2vec_walks(
    G: nx.Graph,
    num_walks: int = 10,
    walk_length: int = 20,
    p: float = 1.0,
    q: float = 1.0,
    seed: int = RANDOM_SEED,
) -> List[List[Any]]:
    """Generate 2nd-order biased random walks (Grover & Leskovec, 2016)."""
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


def train_node2vec_embeddings(
    walks: List[List[Any]],
    dim: int = 16,
    window_size: int = 4,
    n_neg: int = 5,
    epochs: int = 5,
    lr: float = 0.025,
    seed: int = RANDOM_SEED,
) -> Dict[Any, np.ndarray]:
    """Train Skip-Gram with Negative Sampling (SGNS) embeddings from random walks.

    Learns continuous d-dimensional vector representations for all graph nodes.
    """
    rng = np.random.default_rng(seed)
    unique_nodes = sorted(list(set(n for walk in walks for n in walk)))
    node_to_idx = {n: i for i, n in enumerate(unique_nodes)}
    vocab_size = len(unique_nodes)

    # Initialize input and context embeddings
    scale = 0.5 / dim
    W_in = rng.uniform(-scale, scale, (vocab_size, dim)).astype(np.float64)
    W_out = np.zeros((vocab_size, dim), dtype=np.float64)

    # Word frequency for negative sampling (unigram distribution ^ 0.75)
    counts = np.zeros(vocab_size)
    for walk in walks:
        for node in walk:
            counts[node_to_idx[node]] += 1
    unigram = counts ** 0.75
    unigram /= unigram.sum()

    for epoch in range(epochs):
        curr_lr = max(lr * (1.0 - epoch / epochs), 0.001)
        for walk in walks:
            indices = [node_to_idx[n] for n in walk]
            for pos, target_idx in enumerate(indices):
                start = max(0, pos - window_size)
                end = min(len(indices), pos + window_size + 1)
                context_indices = [indices[j] for j in range(start, end) if j != pos]

                for ctx_idx in context_indices:
                    # Positive pair
                    z_pos = np.dot(W_in[target_idx], W_out[ctx_idx])
                    sig_pos = 1.0 / (1.0 + np.exp(-np.clip(z_pos, -15.0, 15.0)))
                    grad_pos = sig_pos - 1.0

                    # Negative samples
                    neg_samples = rng.choice(vocab_size, size=n_neg, p=unigram)
                    neg_dots = np.dot(W_out[neg_samples], W_in[target_idx])
                    sig_negs = 1.0 / (1.0 + np.exp(-np.clip(-neg_dots, -15.0, 15.0)))
                    grad_negs = 1.0 - sig_negs

                    # Gradient updates
                    d_win = grad_pos * W_out[ctx_idx] + np.sum(grad_negs[:, None] * W_out[neg_samples], axis=0)
                    W_out[ctx_idx] -= curr_lr * (grad_pos * W_in[target_idx])
                    W_out[neg_samples] -= curr_lr * (grad_negs[:, None] * W_in[target_idx])
                    W_in[target_idx] -= curr_lr * d_win

    embeddings = {node: W_in[node_to_idx[node]].copy() for node in unique_nodes}
    return embeddings


# =====================================================================
# 3. SPECTRAL GRAPH EMBEDDINGS (Normalized Laplacian)
# =====================================================================

def compute_spectral_graph_embeddings(
    G: nx.Graph,
    dim: int = 16,
    nodes: Optional[List[Any]] = None,
) -> pd.DataFrame:
    """Compute normalized Laplacian spectral embeddings via eigen-decomposition."""
    if nodes is None:
        nodes = list(G.nodes())

    A = nx.to_numpy_array(G, nodelist=nodes, weight="weight")
    deg = A.sum(axis=1)
    deg_inv_sqrt = np.power(deg, -0.5, where=deg > 0)
    deg_inv_sqrt[deg == 0] = 0.0
    D_inv = np.diag(deg_inv_sqrt)
    L_norm = np.eye(len(nodes)) - D_inv @ A @ D_inv

    eigvals, eigvecs = np.linalg.eigh(L_norm)
    # Take non-trivial smallest eigenvectors
    embeddings = eigvecs[:, 1:dim + 1]

    cols = [f"graph_emb_{i}" for i in range(embeddings.shape[1])]
    df_emb = pd.DataFrame(embeddings, columns=cols)
    df_emb["product_ID"] = nodes
    return df_emb


# =====================================================================
# 4. INDUCTIVE KNN NEIGHBOR-FEATURE AGGREGATION
# =====================================================================

class KNNNeighborAggregator:
    """Inductive KNN neighbor feature aggregation with distance-to-similarity weighting.

    Note: Correctly renamed from the legacy 'GraphSAGE' misnomer to reflect actual algorithm.
    """

    def __init__(self, n_neighbors: int = 8):
        self.n_neighbors = n_neighbors
        self.nn: Optional[NearestNeighbors] = None
        self.train_features: Optional[np.ndarray] = None
        self.train_product_ids: Optional[np.ndarray] = None

    def fit(self, X_train: np.ndarray, product_ids: np.ndarray) -> KNNNeighborAggregator:
        """Fit KNN graph strictly on training products to prevent test leakage."""
        self.nn = NearestNeighbors(n_neighbors=min(self.n_neighbors, len(X_train)), metric="cosine")
        self.nn.fit(X_train)
        self.train_features = X_train
        self.train_product_ids = product_ids
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Compute similarity-weighted neighbor aggregations for input nodes."""
        if self.nn is None or self.train_features is None:
            raise ValueError("KNNNeighborAggregator must be fitted before calling transform.")

        distances, indices = self.nn.kneighbors(X)
        # Convert cosine distance d in [0, 2] to similarity s = 1 / (1 + d)
        similarities = 1.0 / (1.0 + distances)
        weights = similarities / np.sum(similarities, axis=1, keepdims=True)

        n_samples, n_features = X.shape
        aggregated = np.zeros((n_samples, n_features), dtype=np.float64)

        for i in range(n_samples):
            nbr_feats = self.train_features[indices[i]]
            aggregated[i] = np.dot(weights[i], nbr_feats)

        return aggregated
