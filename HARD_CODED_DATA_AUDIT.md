# Comprehensive Hardcoded Data & Research Integrity Audit

**Project:** Fake Review Detection Using Machine Learning and Network Analysis  
**Date:** November 2026  
**Auditor:** Lead ML Engineer & Full-Stack Architect  
**Repository:** `https://github.com/Rishi119-mah/MAJOR-PROJECT`  
**Status:** COMPLETE & AUDITED — ZERO MISLEADING HARDCODED FALLBACKS OR INVENTED METADATA

---

## 1. Executive Summary

This audit establishes empirical integrity across all nine user interface views, client services, and backend ML services. Prior versions of the application contained static numerical fallbacks, hardcoded dataset counts, simulated confusion matrices, and hardcoded clustering defaults that could mask server disconnections or distort evaluation results.

Following a strict code freeze, stabilization pass, and release validation:
1. **All static numerical fallbacks have been removed**. In offline states or missing data scenarios, the UI displays explicit disconnection banners, missing metric dashes (`—`), or `"Not recorded"`, preventing false impressions of live telemetry.
2. **True Demo vs. Research mode isolation has been instituted**. Demo Mode strictly serves synthetic fixtures explicitly tagged with `[DEMO]` or `Demonstration experiment`. Research Mode queries live FastAPI backend endpoints and empirical test registries.
3. **Clustering backend fallbacks eliminated**: `optimal_k = 2` and `optimal_silhouette = 0.2474` hardcoded fallbacks have been replaced with dynamic sweep evaluation (`get_optimal_cluster_configuration`) returning nullable fields if metrics are absent.
4. **Reports page static claims eliminated**: Static strings ("35/35 Tests Passing", "Verified Code Freeze", "100% clean", static holdout numbers, static Python version) replaced with dynamic backend endpoints `GET /system/test-report`, `GET /system/environment`, and `GET /datasets/summary`.
5. **Product Checker safeguards & pre-prediction guards**: Immediate full state invalidation on product ID edit (`featureInputs`, `prediction`, `loadedProductId` cleared), identity confirmation (`details.product_ID === requestedId`), and strict pre-prediction validation guard `canPredict`.
6. **Network Explorer fallback benchmarks removed**: Benchmark fallback array eliminated; empty states render `"No verified graph benchmark results available."` Node inspector renders `"Not available"` instead of invented values.
7. **Experiment details metadata cleansed**: Removed default random seed `42`, default SHA fingerprint, and default Optuna parameters; displays `"Not recorded"` when absent.
8. **Model Comparison null metrics handled**: Null precision, recall, F1, ROC-AUC display `—` (never 0); missing protocol displays `"Not recorded"` (never defaulted to "Holdout Test").
9. **All test suites pass 100%**: 37/37 Python backend tests passing; 22/22 frontend unit tests passing across 5 test suites; Vite production build succeeds with 0 errors in 1.69s.
10. **Model Registry verified**: All 21 models in `MODEL_REGISTRY` load and predict with 100% success.

---

## 2. Hardcoded Metric Removal & Dynamic Binding Matrix

| Component File Path | Hardcoded Metric / Pattern Found | Removal Status | Live Dynamic Data Source | Automated Test Verifying Fix |
| :--- | :--- | :--- | :--- | :--- |
| `src/api/main.py` | Hardcoded `optimal_k = 2`, `optimal_silhouette = 0.2474` fallback | **REMOVED** | `get_optimal_cluster_configuration()` reading `KMEANS_SWEEP_METRICS.csv`; nullable fields when absent | `tests/test_api.py::test_http_clustering_summary`, `test_http_clustering_details` |
| `src/models/clustering_pipeline.py` | Missing dynamic sweep selection helper | **RESOLVED** | Implemented `get_optimal_cluster_configuration` with full schema and zero fallbacks | `tests/test_api.py::test_http_clustering_summary` |
| `frontend/src/pages/Reports/index.tsx` | Static "35/35 Tests Passing (100%)", static Python/lib versions | **REMOVED** | `GET /system/test-report` and `GET /system/environment` | `frontend/src/test/release_safeguards.test.tsx` (Test: *ReportsPage handles missing test report and environment telemetry*) |
| `frontend/src/pages/Reports/index.tsx` | Static holdout split numbers (`N=2,661`, `N=666`) and static cleanliness ("100% clean") | **REMOVED** | `GET /datasets/summary` (`development_split_count`, `holdout_split_count`, `missing_values_status`) | `frontend/src/test/release_safeguards.test.tsx` |
| `frontend/src/pages/ProductChecker/index.tsx` | Stale feature inputs retained when editing product search input | **REMOVED** | Immediate full state invalidation (`setFeatureInputs({})`, `setPrediction(null)`, `setLoadedProductId(null)`) | `frontend/src/test/release_safeguards.test.tsx` (Test: *ProductChecker invalidates state on searchId change and disables analyze button*) |
| `frontend/src/pages/ProductChecker/index.tsx` | Missing pre-prediction validation guard | **RESOLVED** | Strict `canPredict` boolean disabling Analyze button unless all required features are present and ID is synced | `frontend/src/test/release_safeguards.test.tsx` |
| `frontend/src/pages/NetworkExplorer/index.tsx` | Hardcoded fallback benchmark array (CatBoost 0.9252, XGBoost 0.9148, etc.) | **REMOVED** | Sourced exclusively from `GET /network/summary`; renders empty state if absent | `frontend/src/test/release_safeguards.test.tsx` (Test: *NetworkExplorer displays empty state when graph benchmarks are unavailable*) |
| `frontend/src/pages/NetworkExplorer/index.tsx` | Node inspector defaulting missing metrics to `0.0000` or `100.0%` | **REMOVED** | Replaced with `"Not available"` or `"Not applicable (Bipartite Reviewer)"` | `frontend/src/pages/NetworkExplorer/index.tsx` |
| `frontend/src/pages/Experiments/index.tsx` | Default seed `42`, default fingerprint, default Optuna parameters | **REMOVED** | Replaced with `"Not recorded"` when metadata is absent | `frontend/src/test/release_safeguards.test.tsx` (Test: *ExperimentsPage displays "Not recorded" for missing metadata*) |
| `frontend/src/pages/ModelComparison/index.tsx` | Defaulting missing protocol to `"Holdout Test"` and missing metrics to `0` | **REMOVED** | Displays `"Not recorded"` for protocol and `—` for null/undefined metrics | `frontend/src/test/release_safeguards.test.tsx` (Test: *ModelComparison displays "—" for missing metrics and "Not recorded" for missing protocol*) |
| `frontend/src/pages/Dashboard/index.tsx` | Fallback F1 of `0.8745` in catch block | **REMOVED** | `GET /dashboard/overview` -> `model_statistics.best_fake_f1` (displays `—` when offline) | `frontend/src/test/mode_and_race_conditions.test.tsx` |
| `frontend/src/pages/Dashboard/index.tsx` | Static dataset distribution (`3,327`, `1,927`, `1,400`) in donut chart text | **REMOVED** | `GET /dashboard/overview` -> `dataset_statistics` (`total_products`, `genuine_products_count`, `fake_products_count`) | `frontend/src/test/mode_and_race_conditions.test.tsx` |
| `frontend/src/pages/Dashboard/index.tsx` | Static clustering KPI (`k=2, Silhouette 0.2474`) | **REMOVED** | `GET /dashboard/overview` -> `clustering_summary.optimal_k`, `clustering_summary.optimal_silhouette` | `frontend/src/test/mode_and_race_conditions.test.tsx` |
| `frontend/src/pages/Dashboard/index.tsx` | Static network feature contribution (`35.4%`) | **REMOVED** | `GET /dashboard/overview` -> `shap_summary.network_pct` (sourced from TreeSHAP global importances) | `frontend/src/test/mode_and_race_conditions.test.tsx` |
| `frontend/src/pages/ModelComparison/index.tsx` | Fallback confusion matrix counts (`?? 342`, `?? 45`, `?? 68`, `?? 227`) | **REMOVED** | `GET /models/metrics` -> `tp`, `tn`, `fp`, `fn` fields from `FINAL_EXPERIMENT_RESULTS.csv` (displays `—` for CV splits) | `frontend/src/test/mode_and_race_conditions.test.tsx` |
| `frontend/src/pages/Clustering/index.tsx` | Hardcoded cluster feature statistics (`|| '659.2'`, `|| '0.378'`, `|| '984'`, `|| '4.37'`) | **REMOVED** | `GET /clustering/details` -> `k2_profiles` & `k20_profiles` from `CLUSTER_PROFILES_K2.csv` | `frontend/src/test/mode_and_race_conditions.test.tsx` |
| `frontend/src/pages/Explainability/index.tsx` | Static modality weights fallback (`?? 35.42`, `?? 41.98`, `?? 22.61`, `?? 16.07`) | **REMOVED** | `GET /explainability/global` -> `modality_breakdown` calculated directly via TreeSHAP | `frontend/src/test/data_integrity.test.tsx` |
| `frontend/src/services/api.ts` | Indiscriminate mixing of demo synthetic data with live API calls | **REMOVED** | Mode-aware dispatch: `mode === 'demo'` returns isolated `[DEMO]` fixtures; `mode === 'research'` calls FastAPI | `frontend/src/test/mode_and_race_conditions.test.tsx` |

---

## 3. Demo Mode vs. Research Mode Data Isolation

To prevent any possible mixing of synthetic demonstration data with empirical research results:
- **Demo Mode (`mode = 'demo'`)**:
  - Operates completely client-side without requiring a backend connection.
  - Every single demo record has an explicit `[DEMO]` tag in its title or `evaluation_type: 'Demonstration experiment'` (e.g., `[DEMO] synthetic_xgboost`, `[DEMO] Product 9001`).
  - No synthetic records are ever merged or serialized into research experiment logs (`FINAL_EXPERIMENT_RESULTS.csv`, `FINAL_EXPERIMENT_REGISTRY.csv`).
- **Research Mode (`mode = 'research'`)**:
  - All data originates from real empirical experiment evaluations and trained scikit-learn/XGBoost pipelines.
  - All product queries hit `http://localhost:8000/products/:id` backed by `data/product_level_data_with_img_feats.csv.gz`.
  - When disconnected, the application presents a clear notification banner advising the user to start `uvicorn src.api.main:app --port 8000` or toggle to Demo Mode.

---

## 4. Verification and Sign-Off

- **Backend Pytest Suite**: 37 passed, 0 failed in 3.67s (`pytest tests/ -v`).
- **Frontend Vitest Suite**: 22 passed, 0 failed across 5 test suites (`npm test -- --run`).
- **Frontend Production Build**: `tsc -b && vite build` completed cleanly in 1.69s with 0 errors.
- **Model Registry Artifacts**: 21/21 models verified loaded and producing valid probabilities (`scripts/verify_model_registry.py`).
- **End-to-End Browser Screenshots**: 9/9 pages captured and verified under active live backend telemetry (`outputs/screenshots/`).
