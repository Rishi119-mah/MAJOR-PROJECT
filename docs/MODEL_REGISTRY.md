# MODEL_REGISTRY.md — Production Model Catalog and Architecture Specifications

## 1. Registered Supervised Pipeline Artifacts

All models are serialized as complete scikit-learn `Pipeline` objects located in `outputs/models/` and registered with strict schema validation in `src/api/main.py`.

| Model Identifier | Artifact File | Modality / Feature Group | Feature Count | Algorithm & Primary Hyperparameters | Preprocessing | Holdout Fake F1 | Holdout ROC-AUC | ECE |
|------------------|---------------|--------------------------|---------------|-------------------------------------|---------------|-----------------|-----------------|-----|
| `review_only_xgboost` | `review_only_xgboost_pipeline.joblib` | Review Behaviors | 12 | `XGBClassifier(n_estimators=100, max_depth=6, lr=0.1)` | None | 0.8124 | 0.8912 | 0.0542 |
| `review_only_random_forest` | `review_only_random_forest_pipeline.joblib` | Review Behaviors | 12 | `RandomForestClassifier(n_estimators=100, max_depth=40)` | None | 0.8065 | 0.8841 | 0.0618 |
| `review_only_catboost` | `review_only_catboost_pipeline.joblib` | Review Behaviors | 12 | `CatBoostClassifier(iterations=200, lr=0.1)` | None | 0.8180 | 0.8950 | 0.0512 |
| `review_only_stacking` | `review_only_stacking_pipeline.joblib` | Review Behaviors | 12 | `StackingClassifier(LR, RF, SVC, XGB -> Meta LR)` | Conditional | 0.8192 | 0.8974 | 0.0498 |
| `network_only_xgboost` | `network_only_xgboost_pipeline.joblib` | Network Centrality | 4 | `XGBClassifier(n_estimators=100, max_depth=6, lr=0.1)` | None | 0.6840 | 0.7621 | 0.0820 |
| `network_only_random_forest` | `network_only_random_forest_pipeline.joblib` | Network Centrality | 4 | `RandomForestClassifier(n_estimators=100, max_depth=40)` | None | 0.6720 | 0.7510 | 0.0894 |
| `network_only_catboost` | `network_only_catboost_pipeline.joblib` | Network Centrality | 4 | `CatBoostClassifier(iterations=200, lr=0.1)` | None | 0.6890 | 0.7680 | 0.0795 |
| `network_only_stacking` | `network_only_stacking_pipeline.joblib` | Network Centrality | 4 | `StackingClassifier(LR, RF, SVC, XGB -> Meta LR)` | Conditional | 0.6910 | 0.7712 | 0.0760 |
| `review_network_xgboost` | `review_network_xgboost_pipeline.joblib` | Review + Network | 16 | `XGBClassifier(n_estimators=100, max_depth=6, lr=0.1)` | None | 0.8520 | 0.9284 | 0.0460 |
| `review_network_random_forest` | `review_network_random_forest_pipeline.joblib` | Review + Network | 16 | `RandomForestClassifier(n_estimators=100, max_depth=40)` | None | 0.8440 | 0.9210 | 0.0520 |
| `review_network_catboost` | `review_network_catboost_pipeline.joblib` | Review + Network | 16 | `CatBoostClassifier(iterations=200, lr=0.1)` | None | 0.8560 | 0.9310 | 0.0435 |
| `review_network_stacking` | `review_network_stacking_pipeline.joblib` | Review + Network | 16 | `StackingClassifier(LR, RF, SVC, XGB -> Meta LR)` | Conditional | 0.8590 | 0.9340 | 0.0410 |
| `review_network_image_xgboost` | `review_network_image_xgboost_pipeline.joblib` | Multimodal All | 28 | `XGBClassifier(n_estimators=100, max_depth=6, lr=0.1)` | None | 0.8642 | 0.9412 | 0.0412 |
| `review_network_image_random_forest` | `review_network_image_random_forest_pipeline.joblib` | Multimodal All | 28 | `RandomForestClassifier(n_estimators=100, max_depth=40)` | None | 0.8580 | 0.9350 | 0.0480 |
| `review_network_image_catboost` | `review_network_image_catboost_pipeline.joblib` | Multimodal All | 28 | `CatBoostClassifier(iterations=200, lr=0.1)` | None | 0.8680 | 0.9430 | 0.0390 |
| `review_network_image_stacking` | `review_network_image_stacking_pipeline.joblib` | Multimodal All | 28 | `StackingClassifier(LR, RF, SVC, XGB -> Meta LR)` | Conditional | 0.8710 | 0.9460 | 0.0385 |
| `tuned_xgboost` | `tuned_xgboost_pipeline.joblib` | Multimodal All | 28 | `XGBClassifier(Optuna-tuned: depth, lr, subsample, colsample)` | None | 0.8745 | 0.9485 | 0.0370 |
| `tuned_random_forest` | `tuned_random_forest_pipeline.joblib` | Multimodal All | 28 | `RandomForestClassifier(Optuna-tuned: estimators, depth, leaf)` | None | 0.8620 | 0.9390 | 0.0440 |
| `tuned_lightgbm` | `tuned_lightgbm_pipeline.joblib` | Multimodal All | 28 | `LGBMClassifier(Optuna-tuned: num_leaves, max_depth, lr)` | None | 0.8710 | 0.9450 | 0.0380 |
| `calibrated_review_network_image_xgboost` | `calibrated_review_network_image_xgboost_pipeline.joblib` | Multimodal All | 28 | `CalibratedClassifierCV(XGBClassifier, method='isotonic', cv=5)` | None | 0.8650 | 0.9420 | 0.0210 |
| `calibrated_review_network_xgboost` | `calibrated_review_network_xgboost_pipeline.joblib` | Review + Network | 16 | `CalibratedClassifierCV(XGBClassifier, method='isotonic', cv=5)` | None | 0.8530 | 0.9290 | 0.0240 |

---

## 2. Graph Neural Networks & Representation Learning

Implemented in `src/graph/gnn_models.py` and benchmarked in `outputs/experiments/GRAPH_ML_BENCHMARK.csv`:

### 2.1 TrainableGraphSAGE
- **Architecture**: Genuine 2-layer Graph Neural Network with analytical backpropagation and Adam optimizer.
- **Aggregation**: Mean neighborhood aggregator with self-connection weight matrix:
  $$h_v^{(l)} = \sigma\left(W_1^{(l)} h_v^{(l-1)} + W_2^{(l)} \cdot \text{MEAN}_{u \in \mathcal{N}(v)}(h_u^{(l-1)})\right)$$
- **Loss**: Binary Cross-Entropy (BCE) with Sigmoid activation.
- **Evaluation**: Inductive vs Transductive node classification benchmarks.

### 2.2 Node2Vec Embeddings
- **Architecture**: Second-order biased random walks ($p=1.0, q=1.0$) followed by Skip-Gram Negative Sampling (SGNS).
- **Optimization**: Numerical gradient descent optimizing pairwise co-occurrence likelihood.
- **Downstream**: Embeddings fed into downstream classifiers.

### 2.3 KNNNeighborAggregator
- **Architecture**: Inductive feature aggregation on bipartite projection:
  $$s(u, v) = \frac{1}{1 + d(u, v)}$$
- **Guarantee**: Fit strictly on development split; test nodes aggregated strictly from training neighbors to prevent data leakage.

---

## 3. Probability Calibration Architecture

Implemented in `src/evaluation/risk_score.py`:

- **Platt Scaling**: Logistic Sigmoid calibration fitted via `CalibratedClassifierCV(method='sigmoid', cv=5)` on development set.
- **Isotonic Regression**: Non-parametric piecewise monotonic calibration fitted via `CalibratedClassifierCV(method='isotonic', cv=5)`.
- **Diagnostics**:
  - Expected Calibration Error (ECE) across 10 uniform probability bins:
    $$\text{ECE} = \sum_{b=1}^{B} \frac{|B_b|}{N} |\text{acc}(B_b) - \text{conf}(B_b)|$$
  - Brier Score Loss: $\text{BS} = \frac{1}{N} \sum_{i=1}^N (p_i - y_i)^2$.

---

## 4. Unsupervised Clustering & Anomaly Profiling

Implemented in `src/experiments/run_clustering_and_robustness.py`:

- **K-Means Sweep ($k=2 \dots 20$)**: Evaluates Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Index.
- **Suspicious Reviewer Rings**: 100 scenario permutations evaluating Recall, Precision, False Positive Rate (FPR), and Specificity.
- **Cluster Profiles**: Output to `outputs/experiments/CLUSTER_PROFILES_K2.csv` and `CLUSTER_PROFILES_K20.csv`.
