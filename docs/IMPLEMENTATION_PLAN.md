# IMPLEMENTATION_PLAN.md — Fake Review Detection System

## Phase Dependencies

```
Phase 0 (Audit) → Phase 1 (Data) → Phase 2 (ML Pipeline) → Phase 3 (Tuning)
                                  → Phase 4 (Graph) → Phase 5 (GNNs)
                                  → Phase 7 (NLP/Image)
                                  → Phase 6 (Temporal)
Phase 2 + 3 → Phase 8 (SHAP)
Phase 4 → Phase 9 (Clustering)
Phase 2 → Phase 10 (Synthetic)
Phase 8 → Phase 11 (Risk Score)
Phase 2 + 8 + 9 → Phase 13 (FastAPI)
Phase 13 → Phase 14 (Hindi/Hinglish)
All → Phase 15 (Testing)
Phase 13 → Phase 16 (Docker)
All → Phase 17 (Report)
```

## Phase 0: Repository Audit ✅ COMPLETE
- [x] Inspect all source files
- [x] Inventory datasets (4 of 5 available)
- [x] Document feature schemas
- [x] Identify issues (15+ found)
- [x] Create PROJECT_CONTEXT.md
- [x] Create REPOSITORY_AUDIT.md
- [x] Preserve original code in legacy/

## Phase 1: Datasets and Preprocessing 🔄
- [x] Dataset inventory and validation
- [x] Cross-reference product IDs across datasets
- [x] Verify label consistency
- [ ] Create data loader module
- [ ] Create preprocessing module
- [ ] Create DATA_DICTIONARY.md
- [ ] Create FEATURE_REGISTRY.md

## Phase 2: Supervised ML Pipeline 🔄
- [ ] Reproduce legacy baseline (exact original behavior)
- [ ] Implement stratified splitting
- [ ] Add DummyClassifier baseline
- [ ] Add RBF SVM
- [ ] Add LightGBM
- [ ] Add CatBoost
- [ ] Add Stacking Ensemble
- [ ] Implement all metrics (fake-class P/R/F1, PR-AUC, etc.)
- [ ] Run Review Only experiments
- [ ] Run Review + Network experiments (MISSING in original)
- [ ] Run Review + Network + Image experiments
- [ ] Run TF-IDF Text experiments
- [ ] Implement 5×2 cross-validation
- [ ] Compare class weights vs SMOTE
- [ ] Save all results to EXPERIMENT_RESULTS.csv

## Phase 3: Hyperparameter Tuning ⏳
- [ ] Install Optuna
- [ ] Smoke test on small subset
- [ ] Tune Random Forest
- [ ] Tune XGBoost
- [ ] Tune LightGBM
- [ ] Tune CatBoost (if beneficial)
- [ ] Build Stacking with tuned base models
- [ ] Compare tuned vs default models
- [ ] Save best parameters

## Phase 4: Network Analysis ⏳
- [ ] Validate existing 4 network features
- [ ] Fix iterrows in graph construction
- [ ] Add typed node IDs (prevent collisions)
- [ ] Implement Betweenness Centrality
- [ ] Implement K-core decomposition
- [ ] Implement Jaccard Reviewer Overlap
- [ ] Implement Louvain Community Detection
- [ ] Create graph feature extraction module
- [ ] BLOCKED: Full UCSD graph requires ~1GB review file

## Phase 5: Node2Vec & GNNs ⏳
- [ ] Install PyTorch Geometric
- [ ] Implement Node2Vec embeddings
- [ ] Implement GraphSAGE
- [ ] Implement GCN
- [ ] Compare: Engineered features vs Node2Vec vs GNN
- [ ] Verify/reproduce GraphSAGE F1=0.822 claim

## Phase 6: Temporal Features ⏳
- [ ] Implement Review Burstiness
- [ ] Implement Review Velocity
- [ ] Implement Rating Entropy
- [ ] Implement Temporal Rating Concentration
- [ ] Chronological train/test split
- [ ] Compare random vs chronological evaluation

## Phase 7: NLP & Image ⏳
- [ ] Validate TF-IDF baseline
- [ ] Install sentence-transformers
- [ ] Implement MiniLM embeddings
- [ ] Compare TF-IDF vs MiniLM
- [ ] Validate existing 12 image-similarity features
- [ ] Compare Review+Network vs Review+Network+Image

## Phase 8: SHAP Explainability ⏳
- [ ] Install shap
- [ ] Global SHAP on best tree model
- [ ] Summary plot
- [ ] Feature group importance
- [ ] Local explanations for selected products
- [ ] Verify/calculate weighted degree importance %

## Phase 9: Clustering & Anomaly Detection ⏳
- [ ] Preserve original K-Means k=20
- [ ] Run K-Means k=2–20 with quality metrics
- [ ] Implement Isolation Forest
- [ ] Implement DBSCAN
- [ ] Implement HDBSCAN
- [ ] Implement Louvain community analysis
- [ ] Separate ground-truth from predicted-fake statistics

## Phase 10: Synthetic Robustness ⏳
- [ ] Build synthetic review-ring generator
- [ ] Vary ring parameters
- [ ] Evaluate detection recall
- [ ] Evaluate false-positive rate
- [ ] Document simulation assumptions

## Phase 11: Risk Score ⏳
- [ ] Calibrate model probabilities
- [ ] Implement 0-100 risk score
- [ ] Implement trust-adjusted rating formula
- [ ] Validate assumptions

## Phase 12: Dashboard ⏳
- [ ] Prepare backend data for Lovable frontend
- [ ] No Streamlit implementation needed

## Phase 13: FastAPI Backend ⏳
- [ ] Create API endpoints
- [ ] Pydantic models
- [ ] CORS, error handling
- [ ] API_CONTRACT.md

## Phase 14: Hindi/Hinglish ⏳
- [ ] Explanation text in Hindi/Hinglish
- [ ] Indian regulatory context

## Phase 15: Testing ⏳
- [ ] Data loading tests
- [ ] Feature creation tests
- [ ] Leakage protection tests
- [ ] Classification tests
- [ ] API tests

## Phase 16: Docker ⏳
- [ ] Dockerfile
- [ ] .dockerignore
- [ ] docker-compose.yml

## Phase 17: Report & PPT ⏳
- [ ] Update presentation
- [ ] LaTeX report
- [ ] IEEE paper draft

