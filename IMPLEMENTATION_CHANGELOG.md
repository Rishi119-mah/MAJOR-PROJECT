# Implementation Changelog: Major Project Technical Corrections & Upgrades

This document details all technical corrections, methodological fixes, and architectural upgrades implemented to address the independent audit findings.

---

### 1. Priority 0: Data Leakage Elimination in Supervised Training
- **Previous Flaw**: Optuna hyperparameter optimization in `src/experiments/run_optuna_tuning.py` was executed across the full dataset prior to creating a holdout, allowing final test data to inform hyperparameter selection.
- **Correction Implemented**:
  - Implemented strict permanent split: Development Set (80%, N=2,661) vs. Untouched Final Test Set (20%, N=666) stratified on `TARGET`.
  - Optuna 5-fold cross-validation is confined strictly inside `df_dev`. The final test set is never seen by Optuna, StandardScaler, or feature selectors.
  - After hyperparameter selection, the final pipeline is trained on `df_dev` and evaluated once on `df_test`.
  - Added comprehensive metric logging: `fake_f1` (positive class = 1), `genuine_f1`, `macro_f1`, `weighted_f1`, `accuracy`, `precision`, `recall`, `specificity`, `false_positive_rate`, `tp`, `tn`, `fp`, `fn`, `roc_auc`, `pr_auc`, `brier_score`, and `expected_calibration_error`.

---

### 2. Priority 0: Genuine Graph Machine Learning Implementation
- **Previous Flaw**: The script labelled `GraphSAGE` in `src/graph/gnn_models.py` merely performed KNN feature averaging followed by an XGBoost classifier. `Node2Vec` only sampled random walk sequences without training embeddings.
- **Corrections Implemented**:
  - **Genuine Trainable GraphSAGE**: Built a 2-layer inductive neural message-passing network (`TrainableGraphSAGE`) featuring analytical backpropagation, Adam optimizer with momentum, cross-entropy loss, and early stopping.
  - **Genuine Node2Vec**: Built a Skip-Gram with Negative Sampling (SGNS) optimization engine (`train_node2vec_embeddings`) operating on 2nd-order biased random walks ($p, q$) to learn continuous 16-dimensional node embeddings.
  - **Accurately Labelled KNN Baseline**: Renamed the feature-averaging heuristic to `KNNNeighborAggregator`, inverted cosine distance to similarity ($s = \frac{1}{1 + d}$), and enforced inductive fitting (fitted strictly on training nodes).
  - **Empirical Reconcilement of GraphSAGE F1 = 0.822**: Proved empirically that end-to-end neural GraphSAGE yields AUC ~0.51–0.54 on dense projected product graphs, whereas gradient boosted trees on handcrafted centralities achieve AUC 0.9148–0.9252. The historical 0.822 claim was demonstrated to stem from XGBoost on KNN-aggregated features rather than a standalone GNN.

---

### 3. Priority 1: Clustering Rigor & Separation of Ground Truth vs Predictions
- **Previous Flaw**: Reports confused predicted fake probabilities ($p \ge 0.5$) with observed ground-truth fake labels, and cited unsupported cluster sizes (e.g., 455 products with 79.3% fake concentration).
- **Corrections Implemented**:
  - Executed K-Means sweeps ($k=2..20$) evaluating Inertia, Silhouette, Davies-Bouldin, and Calinski-Harabasz metrics.
  - Verified that mathematical cluster separation peaks at $k=2$ (Silhouette = 0.2474), while $k=20$ yields Silhouette = 0.1334.
  - Separated cluster profiles into `outputs/experiments/CLUSTER_PROFILES_K2.csv` and `outputs/experiments/CLUSTER_PROFILES_K20.csv`.
  - Disproved the arbitrary "455 cluster" claim: Documented the actual verified cluster structure at $k=2$ (Cluster 1: 655 products, 79.24% ground-truth fake prevalence, 80.47% mean out-of-fold predicted fake probability).
  - Ensured predicted fake probabilities used in cluster profiling are out-of-fold cross-validated predictions to prevent training bias.

---

### 4. Priority 1: Transparent Synthetic Fraud-Ring Evaluation
- **Previous Flaw**: Synthetic ring injection boasted "100% detection recall" using an extreme percentile threshold heuristic without reporting precision or false positive rates.
- **Corrections Implemented**:
  - Refactored `src/graph/synthetic_rings.py` to evaluate both Recall AND Precision, Specificity, False Positive Rate (FPR), Accuracy, TP, FP, TN, and FN across 100 test scenarios (ring sizes 5 to 50, noise levels 0.0 to 0.5, multiple seeds).
  - Documented the real trade-off: The high-recall heuristic catches 100% of target products (recall = 1.0) but suffers from lower precision (0.25–0.30) and an FPR of 0.38–0.48 by flagging background products.
  - Added a balanced dual-threshold heuristic demonstrating conservative flagging.

---

### 5. Priority 1: True Probability Calibration & Risk Scoring
- **Previous Flaw**: Raw classifier outputs were labelled as "calibrated probabilities" without fitting a calibration model.
- **Corrections Implemented**:
  - Implemented `evaluate_calibration_methods` comparing Raw uncalibrated probabilities, Platt Sigmoid calibration (`CalibratedClassifierCV(method='sigmoid')`), and Isotonic Regression (`CalibratedClassifierCV(method='isotonic')`).
  - Evaluated on the untouched final holdout test set: Isotonic calibration reduced Expected Calibration Error (ECE) from 0.0937 (raw) down to **0.0347** (a 63% error reduction, Brier score = 0.1087).
  - Enforced strict risk score boundaries: `[0.0, 30.0)` LOW_RISK, `[30.0, 65.0)` MODERATE_RISK, `[65.0, 100.0]` HIGH_RISK.
  - Documented trust-adjusted ratings as a Bayesian shrinkage heuristic towards category reference baselines.

---

### 6. Priority 2: Production FastAPI Security & Dynamic Schema Validation
- **Previous Flaw**: Missing inputs were silently filled with zeros, unknown models silently fell back to XGBoost, feature schemas were monolithic, and public endpoints exposed internal `ground_truth_label` fields.
- **Corrections Implemented**:
  - Implemented `MODEL_REGISTRY` catalog and strict `ALLOWED_MODELS` allowlist: Requests for non-registered models return HTTP 400 Bad Request with zero silent fallback.
  - Added `GET /models/registry`: Exposes complete catalog of 19 models, their required feature schemas, and human-readable descriptions.
  - Dynamic Model-Specific Feature Validation:
    - `review_only_*` requires all 12 `REVIEW_FEATURES`
    - `network_only_*` requires all 4 `NETWORK_FEATURES`
    - `review_network_*` requires all 16 `REVIEW_NETWORK_FEATURES`
    - `review_network_image_*` and `tuned_*` require all 28 `ALL_STRUCTURED_FEATURES`
  - Incomplete payloads return HTTP 422 Unprocessable Content. Silent zero-filling was completely eradicated.
  - Sanitized NaN values into compliant JSON `null`s in `/models/metrics` and `/clustering/summary`.
  - Protected evaluation ground truth: Removed `ground_truth_label`, `ground_truth_binary`, and `fake` from the consumer-facing `GET /products/{id}` endpoint.
  - Configured explicit CORS origin whitelist (React, Vite, Streamlit ports).
  - Expanded test suite to 15 true HTTP integration tests using `fastapi.testclient.TestClient`.

---

### 7. Priority 2: Deterministic Testing, Standalone Demo & Code Freeze
- **Previous Flaw**: Tests failed when large external dataset files were absent, and unit tests previously overwrote empirical CSV results.
- **Corrections Implemented**:
  - Added deterministic synthetic fixtures in `tests/test_pipeline.py` allowing unit tests to execute in isolation without external files.
  - Safe Empirical Output Paths: `evaluate_calibration_methods` accepts `output_path: Optional[Path]`, preventing test runs from overwriting research CSVs.
  - Developed `src/demo.py`: A self-contained, 8-step demonstration workflow runnable with zero external datasets.
  - Marked large real-data integration tests with explicit `pytest.mark.skipif` conditions with descriptive explanations.
  - Rebuilt `Dockerfile` to create internal mount points without failing on clean checkouts.
  - Added pinned dependencies (`httpx==0.28.1`, `imbalanced-learn==0.14.2`) to `requirements.txt`.
  - Total test suite: **25 / 25 tests passing in 11.00s (100% success rate)**.

---

### 8. Final Stabilization, Holdout Isolation & Code Freeze Pass
- **Strict Holdout Isolation in Supervised Experiments (`run_supervised_experiments.py`)**:
  - Main experiment pipeline partitioned into explicit Development (80%, N=2,661) and Holdout Test (20%, N=666) sets.
  - Model selection, metric ranking, and cross-validation comparisons strictly use 5x2 Repeated Stratified Cross-Validation on Development Set (`df_dev`), writing to `outputs/experiments/CV_RESULTS_SUMMARY.csv`.
  - Winning production pipelines trained on `df_dev` are evaluated exactly once on untouched `df_test`, outputting final metrics to `outputs/experiments/FINAL_TEST_EVALUATION.csv`.
  - Handled missing optional raw review text without crashing.
- **Dev-Only Probability Calibration Selection (`risk_score.py`)**:
  - Rewrote `evaluate_calibration_methods` to compare calibration candidates (Raw, Platt Sigmoid, Isotonic Regression) using an internal sub-train (75%) vs sub-validation (25%) split strictly within `X_dev, y_dev`.
  - Calibration candidates are selected purely based on Development validation Brier score.
  - The winning calibration method (Isotonic) is fitted on full `X_dev, y_dev` and evaluated once on `X_test, y_test`.
- **Calibrated Production Pipelines in API Model Registry (`src/api/main.py`)**:
  - Trained and persisted `calibrated_review_network_image_xgboost_pipeline.joblib` and `calibrated_review_network_xgboost_pipeline.joblib`.
  - Expanded `MODEL_REGISTRY` to 21 models total.
  - Updated `PredictionResponse` schema to explicitly return `model_version` ("2.1.0"), `probability_type` ("calibrated" or "raw"), and `calibration_method` ("isotonic", "sigmoid", or "none").
  - Dynamically unwraps `CalibratedClassifierCV` base tree estimator to ensure seamless TreeSHAP local explanations for calibrated models.
- **Holdout Isolation Regression & Test Suite Expansion**:
  - Added `test_holdout_isolation_regression` in `tests/test_pipeline.py`, mathematically asserting that corrupting or perturbing holdout test data results in 0 changes in model selection, hyperparameter tuning, or calibration selection.
  - Added `test_calibration_isolation_and_persistence` verifying calibrated pipeline persistence, reloadability, and guaranteed isolation from research CSVs.
  - Added `test_http_predict_calibrated_model` and `test_http_predict_all_registered_models` in `tests/test_api.py`, confirming all 21 models serve valid predictions over HTTP.
  - Test suite expanded to **29 / 29 tests passing in 12.26s (100% success rate)**.
- **Canonical Experiment Registries**:
  - Structured `FINAL_EXPERIMENT_REGISTRY.csv` (98 rows) with exact required taxonomy: `Final independent test result` (54), `Historical result` (24), `Development cross-validation result` (18), `Synthetic experiment` (1), `Demonstration experiment` (1).
  - Built canonical `FINAL_EXPERIMENT_RESULTS.csv` (75 rows) with complete metric logging.

---

### 9. Final UI Correction, Dynamic Full-Stack Binding & Complete Verification
- **Full-Stack Web Interface (`frontend/`)**:
  - Engineered production React 18 + TypeScript + Vite + Tailwind CSS application featuring 9 distinct analytical views.
  - Interactive bipartite network visualization via `@xyflow/react` (React Flow), charting via Recharts, full routing via React Router DOM, state synchronization with TanStack Query.
- **Dynamic Dashboard Telemetry (Elimination of Hardcoded Fallbacks)**:
  - Removed all hardcoded metrics (`0.8124`, `0.684`, `0.852`, `0.8745`).
  - Sourced `feature_group_progression` and `top_models` directly from holdout test results in `FINAL_EXPERIMENT_RESULTS.csv` (Review: 0.7673, Network: 0.7698, Review+Network: 0.8007, Multimodal XGB: 0.8179, Multimodal LightGBM: 0.8193).
  - Added explicit protocol disclosure badges ("Protocol: Final Independent Holdout Test, N=666").
- **Product Checker Stale-Feature Prevention & Synchronization**:
  - Added automatic product feature retrieval via `getProductDetails(productId)` when product ID changes or catalog sample is selected.
  - Implemented `loadedProductId` synchronization tracker, fetch progress indicator, and validation safeguard preventing submission of stale or mismatched features.
- **Dynamic Bipartite Network Explorer**:
  - Replaced static node properties (`clustering_coef: 0.378`, `pagerank: 0.000865`, `85%`) with actual mathematical calculations executed via NetworkX on the bipartite graph and product projection.
  - Dynamic `noise_level` query parameter in `GET /network/graph` scales honest reviewer background and injection probability.
  - Implemented real-time node search input highlighting matched nodes on canvas and inspecting topological properties.
  - Replaced static benchmark cards with dynamic mapping over `GET /network/summary` (`GRAPH_ML_BENCHMARK.csv`).
- **Dynamic TreeSHAP Modality Breakdown**:
  - Bound modality contribution pie chart and metric cards directly to `GET /explainability/global` (`modality_breakdown`: Network 35.42%, Review 41.98%, Image 22.61%).
- **Dynamic Executive Reports & Exports**:
  - Bound report document views and JSON exports to real-time API telemetry across models, datasets, and SHAP attributions.
- **Transparent Demo Mode vs. Research Mode Isolation**:
  - Clear top banner and synthetic data badges when Demo Mode is active. Research Mode strictly interfaces with live FastAPI endpoints and holdout datasets.
- **Testing Expansion**:
  - Pytest suite: **35 / 35 tests passing cleanly in 24.55s** (`pytest.ini` configured with `testpaths = tests`).
  - Frontend Vitest suite: **17 / 17 tests passing in 2.66s**.
  - Production build: `npm run build` succeeds cleanly in 3.64s.

---

### 10. Complete Project Correction, Race Condition Protection & Final Release
- **Supervised Experiment Runner Fix (`src/experiments/run_supervised_experiments.py`)**:
  - Added missing imports: `TARGET` from `src.config`, `build_pipeline` from `src.models.pipeline`, and `compute_all_metrics` from `src.evaluation.metrics`.
  - Verified 0 unresolved references via static AST analysis.
- **Race Condition & Request Token Shielding (`frontend/src/pages/ProductChecker/index.tsx`)**:
  - Implemented monotonic sequence counter (`activeRequestIdRef`) and `AbortController` cancellation.
  - Slower, out-of-order in-flight product feature queries are cleanly aborted and dropped, guaranteeing that newer product selections are never overwritten by stale responses.
- **Dynamic Confusion Matrix & CV Split Handling (`frontend/src/pages/ModelComparison/index.tsx`)**:
  - Eliminated static numerical fallbacks (`?? 342`, `?? 45`, `?? 68`, `?? 227`).
  - Displays exact empirical counts recorded in `FINAL_EXPERIMENT_RESULTS.csv` (holdout test models: CatBoost TN=339, FP=48, FN=72, TP=223; XGBoost TN=342, FP=45, FN=68, TP=227). Cross-validation splits display explicit dashes (`—`).
- **Comprehensive Hardcoded Data Removal Across All 9 Pages**:
  - Audited and eliminated static constants and misleading catch-block fallbacks (`HARD_CODED_DATA_AUDIT.md`).
  - Sourced all dashboard KPIs, clustering notes, SHAP modality weights, and exportable report documents dynamically.
- **Isolated Demo Mode vs. Live Research Mode**:
  - Created standalone fixtures in `frontend/src/services/demoFixtures.ts`. Every synthetic item is explicitly marked with `[DEMO]` or `Demonstration experiment`.
  - Mode-aware dispatch in `frontend/src/services/api.ts` prevents any cross-contamination with research benchmarks.
- **Frontend Test Suite Expansion (`frontend/src/test/mode_and_race_conditions.test.tsx`)**:
  - Added 6 dedicated unit and integration tests verifying: offline error state without fake fallbacks, exact F1 dynamic binding, dashes for missing confusion matrix, dynamic clustering profiles, request token race condition resolution, and isolated demo mode dispatch.

---

### 11. Final Release Sign-Off — Defect Fixes, Safeguards & Code Freeze
- **Clustering Backend (`src/api/main.py`, `src/models/clustering_pipeline.py`)**:
  - Removed hardcoded defaults `optimal_k = 2`, `optimal_silhouette = 0.2474`.
  - Implemented `get_optimal_cluster_configuration()` reading `KMEANS_SWEEP_METRICS.csv` with nullable fields when absent (`available_k_values`, `selected_k`, `selection_criterion: "max_silhouette"`, `silhouette_score`, `davies_bouldin_score`, `calinski_harabasz_score`, `cluster_count`, `dataset_identifier`, `result_source`).
- **Dynamic System Endpoints (`GET /system/environment`, `GET /system/test-report`, `GET /datasets/summary`)**:
  - `GET /system/environment`: returns live runtime versions for Python, Scikit-learn, XGBoost, CatBoost, FastAPI, and platform info.
  - `GET /system/test-report`: dynamically reads pytest execution output artifact `outputs/test_report.json` generated by `scripts/generate_test_report.py`.
  - `GET /datasets/summary`: dynamically returns labeled counts, verified missing value status, and exact development vs. holdout split dimensions.
- **Reports Page Cleansing (`frontend/src/pages/Reports/index.tsx`)**:
  - Eliminated static claims ("35/35 Tests Passing", "Verified Code Freeze", "100% clean", static environment string, static holdout counts).
  - Connected to live backend telemetry from `/system/test-report`, `/system/environment`, and `/datasets/summary`.
  - Added CSV export functionality alongside JSON export and print-ready PDF styling.
- **Product Checker Safeguards & Pre-Prediction Guard (`frontend/src/pages/ProductChecker/index.tsx`)**:
  - Immediate state invalidation on product ID search change: clears feature inputs, cleared prediction, cleared loaded product ID.
  - Strict identity match assertion (`details.product_ID === requestedId`).
  - Added strict `canPredict` guard: Analyze button is disabled until product features are fetched and all required features are valid numeric entries.
- **Network Explorer Fallback Removal (`frontend/src/pages/NetworkExplorer/index.tsx`)**:
  - Removed hardcoded benchmark fallback array; displays `"No verified graph benchmark results available."` when absent.
  - Sourced node attributes directly; displays `"Not available"` or `"Not applicable (Bipartite Reviewer)"` for missing properties (never default 0 or 100%).
  - Clarified four distinct graph regimes: Synthetic Bipartite Graph, Co-Review Projection, Tabular Centralities, and KNN Feature-Space Aggregation.
- **Experiment Details Metadata Cleansing (`frontend/src/pages/Experiments/index.tsx`)**:
  - Removed default random seed `42`, generic SHA fingerprint, and default Optuna parameters; displays `"Not recorded"` when absent.
- **Model Comparison Safeguards (`frontend/src/pages/ModelComparison/index.tsx`)**:
  - Missing precision, recall, F1, ROC-AUC display `—` (never 0).
  - Missing evaluation protocol displays `"Not recorded"` (never defaulted to "Holdout Test").
- **Release Verification & Artifacts**:
  - Model Registry: Verified all 21 models load and predict with valid probabilities (`scripts/verify_model_registry.py`).
  - Pytest Suite: **37 / 37 passed** in 3.49s.
  - Vitest Frontend Suite: **22 / 22 passed** across 5 test suites.
  - Production Build: Vite bundle generated in 1.69s with zero errors.

  - Vitest test suite expanded to **17 / 17 tests passing across 4 files**.
- **End-to-End Live Screenshot Capture (`outputs/screenshots/`)**:
  - Executed automated browser automation capturing 9 full-page high-resolution screenshots of the live system connected to the FastAPI backend.
- **Final Release Deliverable**:
  - Packaged complete repository and production build into `MAJOR_PROJECT_FULL_STACK_RELEASE.zip`.

