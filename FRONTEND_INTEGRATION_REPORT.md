# Frontend Full-Stack Integration & Verification Report

**Project Title:** Fake Review Detection Using Machine Learning and Network Analysis  
**Application Version:** 2.1.0  
**Release Date:** November 2026  
**Repository:** `https://github.com/Rishi119-mah/MAJOR-PROJECT`  

---

## 1. System Architecture Overview

The system is delivered as a high-performance, containerized full-stack web application designed for academic evaluation, research review, and production deployment.

```
┌────────────────────────────────────────────────────────┐
│               Frontend Web Application                 │
│      React 18 + TypeScript + Vite + Tailwind CSS       │
│      Recharts (Analytics) + React Flow (Topology)      │
│      TanStack Query + Lucide Icons + Multilingual      │
└───────────────────────────┬────────────────────────────┘
                            │
               REST / JSON (Port 8000)
                            │
┌───────────────────────────▼────────────────────────────┐
│                 FastAPI Backend Service                │
│       Uvicorn ASGI Server • CORS Allowed Origins       │
│     Pydantic Schema Validation • Zero-Filling Guard    │
├────────────────────────────────────────────────────────┤
│                 Machine Learning Layer                 │
│   • Calibrated Multimodal XGBoost Classifier (28 Feats)│
│   • TreeSHAP Feature Attributions & Modality Shares    │
│   • Unsupervised K-Means Cluster Analysis (k=2..20)   │
│   • Bipartite Graph Centrality Engine (NetworkX)       │
└────────────────────────────────────────────────────────┘
```

---

## 2. Nine Production Application Pages

| # | Page Name | Route Path | Core Capabilities & Telemetry | Live Screenshot File |
| :--- | :--- | :--- | :--- | :--- |
| 1 | **Overview Dashboard** | `/dashboard` | Executive KPI cards (N=3,327, 42.1% fake buyers, 21 models, best model F1=0.8193), class balance donut chart, multimodal ablation progression, verified holdout leaderboard. | `outputs/screenshots/01_dashboard.png` |
| 2 | **Product Risk Checker** | `/product-checker` | Real-time product risk inference, trust-adjusted rating calculation, risk tier badge (`HIGH_RISK`, `MODERATE_RISK`, `LOW_RISK`), multilingual explanation generation (English, Hindi, Hinglish), race-condition-protected catalog fetching. | `outputs/screenshots/02_product_checker.png` |
| 3 | **Model Comparison** | `/model-comparison` | Benchmark leaderboard across 75+ evaluated pipelines, feature group & protocol filters, interactive confusion matrix inspector with missing-matrix dash (`—`) support. | `outputs/screenshots/03_model_comparison.png` |
| 4 | **Network Explorer** | `/network-explorer` | Interactive React Flow bipartite graph (reviewer-to-product interactions), injected suspicious coordinated review rings, dynamic node centralities (PageRank, Degree, Coordinated Edge Ratio), ring noise slider. | `outputs/screenshots/04_network_explorer.png` |
| 5 | **Clustering Analysis** | `/clustering` | Unsupervised K-Means sweep curves (Silhouette score, Davies-Bouldin index) across $k=2 \dots 20$, cluster profiles comparison, mathematical peak separation isolation ($k=2$, Silhouette 0.2474, Cluster 1 fake prevalence 79.24%). | `outputs/screenshots/05_clustering.png` |
| 6 | **SHAP Explainability** | `/explainability` | Global TreeSHAP feature importance ranking, modality weight breakdown (Network 35.42%, Review 41.98%, Image 22.61%), causal vs correlational disclaimer. | `outputs/screenshots/06_explainability.png` |
| 7 | **Research Experiments** | `/experiments` | Filterable, paginated registry of all 75 historical and holdout experiments, hyperparameters inspector, CSV export capability. | `outputs/screenshots/07_experiments.png` |
| 8 | **Executive Reports** | `/reports` | Print-ready academic reports, dynamic holdout benchmark table, dataset balance profile, downloadable JSON audit export. | `outputs/screenshots/08_reports.png` |
| 9 | **System Settings & Status** | `/settings` | System environment diagnostics, language switcher (English, Hindi, Hinglish), Demo vs Research Mode toggle, active model selector. | `outputs/screenshots/09_settings.png` |

---

## 3. Data Integrity & Verification Highlights

### Zero Hardcoded Fallbacks
In prior builds, offline network failures were masked by catch-block fallback objects containing static figures (e.g., F1 of 0.8745, static confusion matrices of 342/45/68/227). All nine pages now bind strictly to dynamic API responses. If disconnected, the UI explicitly renders disconnection alerts and missing metric dashes (`—`), preventing misrepresentation of research results.

### True Demo vs. Research Mode Isolation
- **Research Mode**: Queries live FastAPI endpoints at `http://localhost:8000` backed by empirical holdout models and the 3,327 labeled product catalog.
- **Demo Mode**: Dispatches to isolated synthetic fixtures clearly labeled with `[DEMO]` or `Demonstration experiment`. Demo fixtures cannot contaminate research benchmark registries.

### Race Condition & Stale Request Protection
`ProductCheckerPage` includes an active sequence counter `activeRequestIdRef` and an `AbortController`. Rapid user selections cancel prior in-flight HTTP requests, ensuring that delayed responses cannot overwrite newer product selections.

---

## 4. Test Verification Summary

### Backend Unit & Integration Tests (pytest)
- **Suite Command:** `./venv/bin/pytest tests/ -v`
- **Result:** **35 / 35 tests passed (100%)** in 24.55s.
- **Coverage:** API health, datasets summary, models metrics, predictions with 28 features, 12 review features, 4 network features, 422 schema validation, calibration persistence, GraphSAGE neural network, Node2Vec embeddings, zero-filling guard.

### Frontend Unit & Component Tests (vitest)
- **Suite Command:** `cd frontend && npx vitest run`
- **Result:** **17 / 17 tests passed (100%)** in 2.66s across 4 test files:
  - `src/test/components.test.tsx`: 4 tests passed
  - `src/test/data_integrity.test.tsx`: 4 tests passed
  - `src/test/mode_and_race_conditions.test.tsx`: 6 tests passed
  - `src/test/routing.test.tsx`: 3 tests passed

### Production Build Verification
- **Build Command:** `cd frontend && npm run build`
- **Result:** TypeScript compile (`tsc -b`) and Vite production bundle generated cleanly with **0 errors** in 3.64s.

---

## 5. Screenshot Gallery & Verification Checklist

All screenshots are stored in `outputs/screenshots/`:

| File | Size | Visual Elements Verified |
| :--- | :--- | :--- |
| `01_dashboard.png` | 205 KB | FastAPI Online badge, 6 KPI cards, donut chart, feature progression ablation, leaderboard |
| `02_product_checker.png` | 156 KB | Catalog product chips, 28-feature input form, risk tier badge, multilingual explanations |
| `03_model_comparison.png` | 217 KB | Bar chart benchmark, confusion matrix inspector with `—` for CV splits, 75-row table |
| `04_network_explorer.png` | 366 KB | React Flow interactive graph with coordinated rings, node inspector, graph benchmarks |
| `05_clustering.png` | 206 KB | Silhouette & Davies-Bouldin line sweeps ($k=2..20$), K=2 vs K=20 profiles, cluster metrics |
| `06_explainability.png` | 181 KB | TreeSHAP bar chart, modality donut (Review 42%, Network 35%, Image 23%), causal principle |
| `07_experiments.png` | 272 KB | Searchable, paginated 75-experiment table, category dropdown filter, CSV export |
| `08_reports.png` | 251 KB | Academic report generator, holdout benchmark table, dataset profile, JSON export |
| `09_settings.png` | 212 KB | Environment telemetry, model allowlist switcher, multilingual selection, mode toggle |

