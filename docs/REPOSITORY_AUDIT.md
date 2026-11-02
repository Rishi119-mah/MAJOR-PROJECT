# REPOSITORY_AUDIT.md — Complete Audit of fake_review_detection
> Audit Date: 2026-11-02  
> Auditor: Automated AI Analysis  
> Repository: https://github.com/dadepro/fake_review_detection

## 1. File Inventory

### Source Files
| File | Lines | Size | Purpose |
|------|-------|------|---------|
| `code/classification.py` | 128 | 4,575 B | Supervised classification experiments |
| `code/clustering.py` | 166 | 6,305 B | Unsupervised clustering + semi-supervised prediction |
| `README.md` | 30 | 2,615 B | Project description |

### Data Files
| File | Size | Rows | Cols | Labels |
|------|------|------|------|--------|
| `product_level_data_with_img_feats.csv.gz` | 627 KB | 3,327 | 33 | ✅ |
| `product_level_data_text_feats.csv.gz` | 13.2 MB | 3,327 | 1,002 | ✅ |
| `product_level_data_without_img_feats.csv.gz` | 325 KB | 3,408 | 18 | ✅ |
| `UCSD_product_level_data.csv.gz` | 7.9 MB | 64,585 | 21 | ❌ |
| `UCSD_home_and_kitchen_reviews.csv.gz` | ~1 GB | ? | ? | ❌ NOT PRESENT |

### Missing Files
- `UCSD_home_and_kitchen_reviews.csv.gz` — required for clustering.py
- `Fake_Review_Detection_Review1_Final.pptx` — project presentation
- No `requirements.txt` in original repo (we generated one)
- No `.gitignore`
- No tests
- No CI/CD configuration
- No Docker configuration

---

## 2. classification.py — Detailed Audit

### 2.1 Data Loading (Lines 14–16)
```python
df = pd.read_csv('product_level_data_with_img_feats.csv.gz')
df_text = pd.read_csv('product_level_data_text_feats.csv.gz')
```
**Issues:**
- ⚠️ Relative path — expects execution from `data/` directory
- ⚠️ No error handling for missing files
- ⚠️ No data validation after loading

### 2.2 Feature Definitions (Lines 19–26)
- 12 review features — **CORRECT**, matches dataset columns
- 4 network features — **CORRECT**, matches dataset columns  
- 12 image similarity features — **CORRECT**, matches dataset columns
- ✅ Feature names are consistent with dataset schemas

### 2.3 model_building Function (Lines 33–49)
```python
def model_building(X_train, y_train, X_test, y_test, model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    probs = model.predict_proba(X_test)[:,1]
```
**Issues:**
- ⚠️ `predict_proba` called twice (line 39 and line 43) — redundant computation
- ⚠️ Prints "TN, TP" but actually reports class-0 recall and class-1 recall
- ⚠️ Uses `weighted` F1, not fake-class-specific F1
- ❌ Missing: precision, PR-AUC, macro F1
- ❌ Missing: confusion matrix export/save
- ❌ Missing: training time measurement
- ❌ Missing: model persistence

### 2.4 classification_results Function (Lines 51–100)

**Train-Test Split (Line 61):**
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
```
- ❌ **No stratification** — `stratify=y` is missing. With 42% positive class, this may not cause severe issues but is methodologically incorrect.

**Preprocessing (Lines 64–66):**
```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```
- ✅ Correctly fits on train, transforms test — no data leakage
- ⚠️ Scales ALL features including for tree-based models (RF, XGBoost) — unnecessary but not harmful

**Models:**

| Model | Config | Issues |
|-------|--------|--------|
| LogisticRegression | `max_iter=400` | ✅ Reasonable |
| RandomForest | `n_estimators=100, max_depth=40, min_samples_leaf=3, min_samples_split=6, max_features='auto'` | ⚠️ `max_features='auto'` is deprecated in newer sklearn (maps to 'sqrt') |
| SVC | `kernel='linear', probability=True` | ❌ **Should be RBF** per project spec. Linear SVM is a different algorithm. |
| XGBClassifier | Default params | ⚠️ No hyperparameter tuning |

**Experiments Run (Lines 102–125):**

| # | Name | Features | Dataset |
|---|------|----------|---------|
| 1 | Review Features | 12 review | df (3,327) |
| 2 | Image Features | 12 image sim | df (3,327) |
| 3 | Network Features | 4 network | df (3,327) |
| 4 | Top-2 Network | eigenvector_cent, clustering_coef | df (3,327) |
| 5 | All Features | 12 review + 12 image + 4 network | df (3,327) |
| 6 | All Text | 1,000 TF-IDF + fake + product_ID → drops ID/label | df_text (3,327) |

**Missing Experiments:**
- ❌ Review + Network (without Image)
- ❌ Review + Network + Image + Text combined
- ❌ Cross-validation experiments
- ❌ Class-imbalance handling experiments

### 2.5 Reported Metrics
- AUC (ROC-AUC)
- Accuracy
- "TN" (actually class-0 recall = specificity)
- "TP" (actually class-1 recall = sensitivity)  
- Weighted F1

**Missing Metrics:**
- ❌ Fake-class Precision
- ❌ Fake-class F1
- ❌ Macro F1
- ❌ PR-AUC (Average Precision)
- ❌ Training time
- ❌ Inference time
- ❌ Brier score
- ❌ Confidence intervals

---

## 3. clustering.py — Detailed Audit

### 3.1 Data Loading (Lines 9–22)
```python
reviews = pd.read_csv('UCSD_home_and_kitchen_reviews.csv.gz')  # ~1GB, NOT PRESENT
df_ucsd = pd.read_csv('UCSD_product_level_data.csv.gz')
df_ours = pd.read_csv('product_level_data_without_img_feats.csv.gz')
```
- ❌ **BLOCKED**: `UCSD_home_and_kitchen_reviews.csv.gz` is not available locally

### 3.2 weighted_projected_graph Function (Lines 31–54)
- Custom re-implementation of NetworkX's `bipartite.weighted_projected_graph`
- Uses simple shared-reviewer count as edge weight
- ✅ Correct algorithm logic
- ⚠️ Potentially slow for large graphs (O(n²) in worst case)

### 3.3 obtain_network_features Function (Lines 56–81)
```python
B.add_edges_from([(row['reviewer_ID'], row['product_ID']) for idx, row in reviews.iterrows()])
```
- ❌ **`iterrows` on potentially millions of rows** — extremely slow
- Should use vectorized `zip(reviews['reviewer_ID'], reviews['product_ID'])`
- ⚠️ **Node ID collision risk**: If any reviewer_ID equals a product_ID, they would be merged into a single node in the bipartite graph. The code uses raw IDs without type prefixing.

### 3.4 Clustering Configuration (Lines 118–122)
```python
X = scaling_data(df_ucsd, features_to_use)
k = 20
method = KMeans(n_clusters=k, random_state=42).fit(X)
```
- ⚠️ Fixed k=20 — no elbow analysis, no silhouette evaluation
- ⚠️ No exploration of k range (k=2–20)
- ⚠️ StandardScaler applied to all features before K-Means — correct practice

### 3.5 Per-Cluster Classification (Lines 126–143)
```python
model = RandomForestClassifier(random_state=42, n_estimators=1200, ...)
```
- ⚠️ **Different RF config** than classification.py (1200 trees vs 100 trees)
- ⚠️ Scaler is commented out (lines 89–91) — inconsistent with classification.py
- ⚠️ `p_fake` is **model-predicted probability**, NOT ground-truth label
- ❌ UCSD products have NO ground-truth labels — cannot validate predictions

### 3.6 Results Aggregation (Lines 146–165)
```python
'p_fake': lambda x: (x >= 0.5).sum()
```
- ⚠️ Counts products with predicted p_fake ≥ 0.5 — this is thresholded model output
- ⚠️ Reported alongside z-scored features — mixed units in final table
- ❌ No cluster quality metrics (silhouette, Davies-Bouldin)
- ❌ No visualization

---

## 4. Data Quality Assessment

### 4.1 Data Cleanliness
| Check | Result |
|-------|--------|
| Missing values | ✅ Zero across all datasets |
| Duplicate IDs | ✅ Zero across all datasets |
| Infinite values | ✅ Zero across all datasets |
| Label consistency | ✅ Verified across img/noimg/text for shared products |
| Feature consistency | ✅ Verified for shared features between img and noimg |

### 4.2 Population Differences
- `without_img_feats` has 3,408 products (1,936 genuine, 1,472 fake = 43.19%)
- `with_img_feats` has 3,327 products (1,927 genuine, 1,400 fake = 42.08%)
- **81 products** in noimg lack image features (9 genuine lost, 72 fake lost)
- The image-available subset is slightly less balanced

### 4.3 UCSD Dataset Special Features
The UCSD dataset has 4 extra columns not in the labelled datasets:
- `degree` (unweighted degree, vs `w_degree` weighted)
- `w_pagerank` (weighted pagerank, vs `pagerank`)
- `w_clustering_coef` (weighted clustering coefficient)
- `w_eigenvector_cent` (weighted eigenvector centrality)

The clustering code uses `review_features + network_features` but the network feature names in UCSD (`pagerank`, `w_degree`, `clustering_coef`, `eigenvector_cent`) match the labelled data feature names. The additional weighted/unweighted variants are unused in the current code.

---

## 5. Reproducibility Issues

| # | Issue | Severity | Fix |
|---|-------|----------|-----|
| 1 | Relative data paths — scripts must run from `data/` | Medium | Use project-root-relative paths |
| 2 | No random seed for full pipeline | Low | Seeds exist for models/split but not for potential shuffling |
| 3 | No requirements.txt in original repo | Medium | Generated locally |
| 4 | No `.gitignore` | Low | Add standard Python gitignore |
| 5 | `max_features='auto'` deprecated | Low | Replace with `'sqrt'` |
| 6 | No experiment logging | High | Add structured experiment registry |
| 7 | No model checkpointing | High | Add joblib persistence |
| 8 | No output files generated | High | Add CSV/JSON result exports |

---

## 6. Missing Components (vs Project Spec)

| Component | Status |
|-----------|--------|
| LightGBM | ❌ Not implemented |
| CatBoost | ❌ Not implemented |
| Stacking Ensemble | ❌ Not implemented |
| RBF SVM | ❌ Not implemented (only Linear SVM) |
| DummyClassifier baseline | ❌ Not implemented |
| Cross-validation | ❌ Not implemented |
| Hyperparameter tuning | ❌ Not implemented |
| SMOTE | ❌ Not implemented |
| Node2Vec | ❌ Not implemented |
| GraphSAGE | ❌ Not implemented |
| GCN | ❌ Not implemented |
| SHAP | ❌ Not implemented |
| Temporal features | ❌ Not implemented |
| MiniLM embeddings | ❌ Not implemented |
| Streamlit dashboard | ❌ Not implemented |
| FastAPI backend | ❌ Not implemented |
| Docker | ❌ Not implemented |
| Tests | ❌ Not implemented |
| Risk scoring | ❌ Not implemented |
| Trust-adjusted rating | ❌ Not implemented |
| Isolation Forest | ❌ Not implemented |
| DBSCAN/HDBSCAN | ❌ Not implemented |
| Louvain community | ❌ Not implemented |
| Betweenness centrality | ❌ Not implemented |
| K-core decomposition | ❌ Not implemented |
| Synthetic review-ring | ❌ Not implemented |

---

## 7. Summary of Required Actions

### Immediate (Phase 1–3)
1. Preserve original code as legacy baseline
2. Fix stratified splitting
3. Add RBF SVM
4. Add LightGBM, CatBoost, Stacking
5. Add missing experiments (Review+Network)
6. Add comprehensive metrics
7. Add cross-validation
8. Add hyperparameter tuning
9. Add DummyClassifier baseline
10. Add class imbalance handling

### Medium-term (Phase 4–9)
11. Fix iterrows in graph construction
12. Add typed node IDs for bipartite graph
13. Implement additional graph features
14. Implement Node2Vec, GraphSAGE, GCN
15. Implement SHAP
16. Implement temporal features
17. Implement MiniLM
18. Improve clustering pipeline
19. Add anomaly detection

### Long-term (Phase 10–17)
20. Synthetic robustness testing
21. Risk scoring
22. FastAPI backend
23. Docker deployment
24. Academic report
25. Updated PPT

