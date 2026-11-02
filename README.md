# Fake Review Detection Using Machine Learning, Network Analysis & Explainable AI

> Based on and extending: *"Detecting Fake Review Buyers Using Network Structure: Direct Evidence from Amazon"* (He et al., 2022).

---

## 1. Quickstart

### Environment Setup
```bash
# Activate existing virtual environment
source venv/bin/activate

# Install all dependencies (if setting up fresh)
pip install -r requirements.txt
```

### Run Tests
```bash
pytest tests/
```
*(All 10 unit and integration tests passing: data loading, metrics, risk scoring, synthetic graph generation, FastAPI routes).*

---

## 2. Running Experiments

### Supervised Learning Suite (Legacy + Improved + 5x2 CV)
```bash
python src/experiments/run_supervised_experiments.py
```
Outputs:
- `outputs/experiments/EXPERIMENT_RESULTS.csv` (All 68 experiments)
- `outputs/experiments/CV_RESULTS_SUMMARY.csv` (5x2 Repeated Stratified CV)
- `outputs/models/` (Trained pipelines for RF, XGBoost, CatBoost, Stacking)

### Optuna Hyperparameter Tuning
```bash
python src/experiments/run_optuna_tuning.py
```
Outputs:
- `outputs/experiments/BEST_HYPERPARAMETERS.json`
- `outputs/models/tuned_*_pipeline.joblib`

### SHAP Explainability & Network Importance
```bash
python src/experiments/run_shap_experiments.py
```
Outputs:
- `outputs/figures/shap_summary_plot.png`
- `outputs/experiments/SHAP_FEATURE_IMPORTANCE.csv`
- Global network weight: **35.42%** (clustering coefficient: 16.07%, eigenvector centrality: 15.38%, weighted degree: 2.88%).

### Clustering, Anomaly Detection & Synthetic Ring Robustness
```bash
python src/experiments/run_clustering_and_robustness.py
```
Outputs:
- `outputs/experiments/KMEANS_SWEEP_METRICS.csv` ($k=2..20$)
- `outputs/experiments/CLUSTER_ANALYSIS_SUMMARY.csv` (Ground truth vs predicted fake rates)
- `outputs/experiments/SYNTHETIC_RING_ROBUSTNESS.csv` (100% recall on injected fraud rings)

### Graph ML & GraphSAGE Benchmark
```bash
python src/experiments/run_gnn_experiments.py
```
Outputs:
- `outputs/experiments/GRAPH_ML_BENCHMARK.csv` (Handcrafted network features vs Spectral embeddings vs GraphSAGE aggregations vs PPT claim).

---

## 3. Running the FastAPI Backend
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger docs: [http://localhost:8000/docs](http://localhost:8000/docs)

Key endpoints:
- `GET /health`: Healthcheck & version
- `GET /datasets/summary`: Statistics on labelled & unlabelled datasets
- `GET /models/metrics`: Real-time benchmark registry
- `POST /predict/product`: Product inference with 0–100 risk score, trust-adjusted rating, and multilingual explanations (English, Hindi, Hinglish)
- `GET /clustering/summary`: Cluster profiles with fake buyer concentrations

---

## 4. Docker Deployment
```bash
# Build and run with Docker Compose
docker-compose up --build -d

# Check health
curl http://localhost:8000/health
```

---

## 5. Documentation Directory (`docs/`)
- `docs/PROJECT_CONTEXT.md`: Persistent system specifications and environment context.
- `docs/REPOSITORY_AUDIT.md`: In-depth audit of original reproduction code and datasets.
- `docs/ACADEMIC_RESEARCH_REPORT.md`: Comprehensive academic major project report.
- `docs/report.tex`: IEEE-formatted LaTeX research paper draft.
- `docs/API_CONTRACT.md`: Complete OpenAPI specifications for Lovable frontend integration.
- `docs/MODEL_REGISTRY.md`: Catalog of all 9 classifiers, graph architectures, and explainers.
- `docs/DATA_DICTIONARY.md`: Schema, types, and descriptions for all 33+ features.
- `docs/FINAL_PROJECT_STATUS.md`: Empirical verification ledger for all 17 phases.
