# TASK_BOARD.md — Execution Board

## 🟢 Completed & Empirically Verified
| Task | Phase | Output / Artifact |
|---|---|---|
| Complete Repository Audit | 0 | `docs/REPOSITORY_AUDIT.md`, `docs/PROJECT_CONTEXT.md` |
| Dataset Schema & Overlap Audit | 1 | `docs/DATA_DICTIONARY.md`, `docs/FEATURE_REGISTRY.md` |
| Exact Legacy Baseline Reproduction | 2 | `legacy/`, 24 legacy experiments logged in `EXPERIMENT_RESULTS.csv` |
| Improved Stratified Supervised Suite | 2 | 9 models x 4 feature combos logged in `EXPERIMENT_RESULTS.csv` |
| 5x2 Repeated Stratified CV | 2 | 19 configurations summarized in `CV_RESULTS_SUMMARY.csv` |
| Optuna Hyperparameter Tuning | 3 | `outputs/experiments/BEST_HYPERPARAMETERS.json`, tuned models in `outputs/models/` |
| Collision-Free Graph Reconstruction | 4 | `src/graph/graph_builder.py` |
| Graph Representation Benchmark | 5 | Handcrafted vs Spectral vs GraphSAGE vs PPT claim in `GRAPH_ML_BENCHMARK.csv` |
| Temporal Features & Proxies | 6 | `src/features/temporal.py` (velocity, burstiness, entropy) |
| Multimodal Feature Progressions | 7 | Review (0.877 AUC) -> Review+Network (0.914 AUC) -> Review+Net+Img (0.920 AUC) |
| TreeSHAP Explainability & Network Weight | 8 | `outputs/figures/shap_summary_plot.png`, 35.42% network importance verified |
| K-Means Sweep ($k=2..20$) & Isolation Forest | 9 | `KMEANS_SWEEP_METRICS.csv`, `CLUSTER_ANALYSIS_SUMMARY.csv` |
| Synthetic Review-Ring Robustness | 10 | 100% recall verified in `SYNTHETIC_RING_ROBUSTNESS.csv` |
| 0-100 Risk Score & Trust Adjusted Rating | 11 | `src/evaluation/risk_score.py` |
| Frontend API Contract for Lovable UI | 12 | `docs/API_CONTRACT.md` |
| Production FastAPI Microservice | 13 | `src/api/main.py` |
| Localization (Hindi/Hinglish) & BIS IS 19000 | 14 | `src/utils/regulations.py`, multilingual explanations in API |
| Automated Pytest Test Suite | 15 | 10/10 tests passing in `tests/test_pipeline.py` & `tests/test_api.py` |
| Docker Containerization | 16 | `Dockerfile`, `docker-compose.yml`, `.dockerignore` |
| Academic Research Report & IEEE Draft | 17 | `docs/ACADEMIC_RESEARCH_REPORT.md`, `docs/report.tex` |

---

## 🟡 External Dataset Note
| Item | Details |
|---|---|
| UCSD Review Level Data (~1GB) | `UCSD_home_and_kitchen_reviews.csv.gz` is hosted on Dropbox. Our architecture cleanly decouples this file, allowing all machine learning, graph representation, explainability, and API capabilities to operate on the 4 locally verified datasets. |
