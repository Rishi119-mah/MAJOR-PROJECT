# MODEL_REGISTRY.md — Model Catalog and Specifications

## Supervised Classifiers

| Model ID | Class / Library | Primary Hyperparameters | Preprocessing | Notes |
|----------|-----------------|-------------------------|---------------|-------|
| `dummy` | `DummyClassifier` (sklearn) | `strategy="most_frequent"` | None | Zero-intelligence baseline |
| `logistic_regression` | `LogisticRegression` (sklearn) | `max_iter=1000, solver='lbfgs'` | `StandardScaler` | Linear parametric baseline |
| `random_forest` | `RandomForestClassifier` (sklearn) | `n_estimators=100, max_depth=40, min_samples_split=6, min_samples_leaf=3, max_features='sqrt'` | None | Primary tree baseline (Paper uses `max_features='auto'`) |
| `svm_linear` | `SVC(kernel='linear')` (sklearn) | `probability=True` | `StandardScaler` | Legacy baseline model |
| `svm_rbf` | `SVC(kernel='rbf')` (sklearn) | `C=1.0, gamma='scale', probability=True` | `StandardScaler` | Required non-linear kernel baseline |
| `xgboost` | `XGBClassifier` (xgboost) | `n_estimators=100, max_depth=6, learning_rate=0.1, eval_metric='logloss'` | None | Primary gradient boosting baseline |
| `lightgbm` | `LGBMClassifier` (lightgbm) | `n_estimators=100, learning_rate=0.1, verbose=-1` | None | Fast gradient boosting |
| `catboost` | `CatBoostClassifier` (catboost) | `iterations=200, learning_rate=0.1, verbose=0` | None | High-performance gradient boosting |
| `stacking` | `StackingClassifier` (sklearn) | Base: LR, RF, SVC RBF, XGB; Meta: LogisticRegression | Conditional per base estimator | Ensemble meta-model |

---

## Graph Models & Embeddings

| Model ID | Method / Architecture | Input Data | Target | Status |
|----------|----------------------|------------|--------|--------|
| `graph_features_xgb` | Handcrafted Graph Features + XGBoost | 4 original + advanced graph features | Binary `fake` | Implemented |
| `node2vec_xgb` | Node2Vec Random Walk Embeddings (d=64) + XGBoost | Product projection graph | Binary `fake` | In Implementation |
| `graphsage` | GraphSAGE (2-layer neighborhood aggregation) | Product projection graph | Binary `fake` | In Implementation |
| `gcn` | Graph Convolutional Network (2-layer GCN) | Product projection graph | Binary `fake` | In Implementation |

---

## Unsupervised & Clustering Models

| Model ID | Library | Configuration | Target / Metric | Status |
|----------|---------|---------------|-----------------|--------|
| `kmeans_k20` | `KMeans` (sklearn) | `k=20, random_state=42` | Paper baseline cluster analysis | Preserved |
| `kmeans_sweep` | `KMeans` (sklearn) | `k=2..20` | Silhouette, Davies-Bouldin, Calinski-Harabasz | In Implementation |
| `isolation_forest` | `IsolationForest` (sklearn) | `contamination=0.1, random_state=42` | Anomaly score | In Implementation |
| `dbscan` | `DBSCAN` (sklearn) | `eps=adaptive, min_samples=5` | Density clusters & noise | In Implementation |

---

## Explainability Models

| Model ID | Framework | Target Model | Explanations Generated |
|----------|-----------|--------------|------------------------|
| `tree_shap` | `shap.TreeExplainer` | Best Tree Model (XGBoost / Random Forest) | Global feature importance summary, local force & waterfall plots |
| `kernel_shap` | `shap.KernelExplainer` | Any black-box model (Stacking / GNN) | Local sample explanations |

