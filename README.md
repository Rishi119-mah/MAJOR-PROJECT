# Fake Review Detection Using Machine Learning and Network Analysis

> Final Year Major Project • Academic Engineering Release  
> Based on and extending: *"Detecting Fake Review Buyers Using Network Structure: Direct Evidence from Amazon"* (He et al., 2022).  
> **Target Repository**: [https://github.com/Rishi119-mah/MAJOR-PROJECT](https://github.com/Rishi119-mah/MAJOR-PROJECT)  

---

## 1. Project Overview
This project presents an end-to-end Machine Learning, Graph Analytics, and Explainable AI system designed to identify **products whose sellers actively purchase fraudulent reviews** on e-commerce platforms like Amazon. Unlike conventional review-level classifiers that label individual review text in isolation, this system classifies entities at the **product level** by fusing behavioral review statistics, reviewer-product network topology, and product image similarity embeddings. The platform delivers a production-grade FastAPI REST service paired with a modern React 18 / TypeScript interactive analytical web dashboard.

---

## 2. Problem Statement
Online product reviews heavily dictate consumer purchasing choices and search ranking algorithms. Unscrupulous sellers exploit this dynamic by commissioning fake positive reviews through coordinated incentivized buyer groups (e.g., private messaging rings, PayPal refund schemes). Standard text-based NLP filters struggle against this practice because paid reviewers write natural-sounding, verified-purchase reviews. Detecting organized fake review manipulation requires modeling the **structural network relations** between reviewers and products alongside behavioral velocity metrics and product imagery.

---

## 3. Project Objectives
1. **Product-Level Classification**: Predict the probability $p(\text{fake buyer} \mid \mathbf{x})$ that an e-commerce product is associated with organized fake-review acquisition.
2. **Network Structure Integration**: Construct reviewer-product bipartite graphs and projected co-review networks to compute structural centralities (PageRank, Degree, Clustering Coefficient, Eigenvector Centrality).
3. **Multimodal Fusion**: Systematically benchmark feature modality progression: Review-Only $\to$ Network-Only $\to$ Review+Network $\to$ Multimodal (Review + Network + Image Embeddings).
4. **Leakage-Free Validation**: Enforce a strict 80% Development Split ($N=2,661$) with 5x2 Repeated Stratified Cross-Validation, evaluating final production models exactly once on an untouched 20% Holdout Test Split ($N=666$).
5. **Calibrated Risk Scoring & Explainability**: Implement Isotonic probability calibration, a bounded 0–100 Continuous Risk Score, Bayesian trust-adjusted star ratings, and multilingual TreeSHAP feature attributions (English, Hindi, and Hinglish).
6. **Full-Stack Deployment**: Deliver a secured FastAPI backend and an interactive 9-page React dashboard featuring interactive React Flow bipartite network exploration.

---

## 4. Technologies Used
- **Backend & ML**: Python 3.14 / 3.11, FastAPI, Uvicorn, Scikit-learn, XGBoost, CatBoost, LightGBM, NetworkX, PyTorch, SHAP, Joblib, NumPy, Pandas.
- **Frontend & UI**: React 18, TypeScript, Vite, Tailwind CSS, Recharts, React Flow (`@xyflow/react`), TanStack Query, React Router DOM, Lucide Icons.
- **Testing & QA**: Pytest, Vitest, Testing Library (`@testing-library/react`), Headless Chromium.
- **Containerization**: Docker, Docker Compose.

---

## 5. System Architecture
```mermaid
flowchart TD
    subgraph Data Layer
        A[Amazon Product Catalog] --> B[Feature Extraction Pipeline]
        B --> C1[12 Review Dynamics Features]
        B --> C2[4 Graph Centrality Features]
        B --> C3[12 Image Cosine Similarities]
    end

    subgraph Modeling & Calibration
        C1 & C2 & C3 --> D[Multimodal Vector X: 28 dims]
        D --> E[Supervised Ensemble Classifiers]
        E -->|Raw Logits| F[Isotonic Probability Calibration]
        F --> G[Calibrated Probability p_fake]
    end

    subgraph Scoring & Explainability
        G --> H1[0-100 Product Risk Score]
        G --> H2[Bayesian Trust-Adjusted Rating]
        G --> H3[TreeSHAP Multilingual Explainer]
    end

    subgraph Presentation Layer
        H1 & H2 & H3 --> I[FastAPI REST Backend: Port 8000]
        I --> J[React 18 / TypeScript Web Dashboard: Port 5173]
    end
```

---

## 6. Machine-Learning Methodology
To guarantee academic integrity and prevent data snooping:
- **Zero-Leakage Split**: Stratified 80% Development Set ($N=2,661$) and 20% Final Holdout Test Set ($N=666$).
- **Dev-Only Model Tuning**: All hyperparameter optimization (Optuna Bayesian search) and probability calibration candidate selections are strictly confined within the 80% Dev split.
- **Ensemble Algorithms**: Evaluated XGBoost, LightGBM, CatBoost, Random Forest, Logistic Regression, Support Vector Machines, and Stacking Ensembles.
- **Decision Metric**: Optimized on Minority Class F1-Score (**Fake F1**) and Area Under ROC (**ROC-AUC**).

---

## 7. Dataset Description
- **Catalog Size**: 3,327 labeled e-commerce products from Amazon Electronics.
- **Class Balance**: 852 positive products (25.61% confirmed fake-review buyers) and 2,475 genuine products (74.39%).
- **Verification**: Zero missing values across all 28 structured feature dimensions.
- **Privacy & Protection**: Public API endpoints sanitize and strip ground-truth training labels (`TARGET`) to prevent evaluation leakage.

---

## 8. Feature Engineering
The feature vector consists of 28 engineered dimensions across three modalities:
1. **Review Dynamics (12 features)**: `n_of_reviews`, `avg_review_rating`, `avg_days_between_reviews`, `stdev_days_between_reviews`, `max_days_between_reviews`, `min_days_between_reviews`, `share_helpful_reviews`, `share_1star`, `share_5star`, `share_photo`, `std_review_len`, `mean_tfidf_review_body`.
2. **Network Centrality (4 features)**: `pagerank`, `w_degree`, `clustering_coef`, `eigenvector_cent`.
3. **Product Image Similarity (12 features)**: ResNet cosine similarity distributions (`min_sim`, `max_sim`, `mean_sim`, `std_sim`, `pct_sim_above_80`, `pct_sim_above_90`, etc.).

---

## 9. Reviewer-Product Network Analysis
Reviewer interactions are represented as a bipartite graph $B = (U, V, E)$, where $U$ is the set of reviewers, $V$ is the set of products, and edge $(u, v) \in E$ denotes a review.
- **Bipartite Projection**: Projected co-review product graphs weight edges by shared reviewer counts.
- **Graph Metrics**: Coordinated fraud rings produce abnormally dense localized subgraphs, yielding high weighted degree and anomalous clustering coefficients.

---

## 10. Graph-Learning Approaches
We benchmarked handcrafted centralities against graph embedding and neural GNN representations:
1. **Tabular Centralities + CatBoost**: ROC-AUC **0.9252**, Fake F1 **0.7880**.
2. **Tabular Centralities + XGBoost**: ROC-AUC **0.9148**, Fake F1 **0.8007**.
3. **KNN Feature-Space Aggregation + XGBoost**: ROC-AUC **0.9133**, Fake F1 **0.7895**.
4. **Genuine Node2Vec (Skip-Gram SGNS $d=16$) + XGBoost**: ROC-AUC **0.9002**, Fake F1 **0.7856**.
5. **Trainable GraphSAGE (2-layer Neural GNN)**: ROC-AUC **0.5101**, Fake F1 **0.4456**.  
*(Finding: Handcrafted topological centralities outperform pure neural GNNs on dense projected bipartite graphs due to severe edge heterophily).*

---

## 11. Clustering & Unsupervised Anomaly Detection
- **K-Means Sweep ($k=2 \dots 20$)**: Mathematical cluster separation reaches its global peak at $k=2$ with a Silhouette score of **0.2474** (Davies-Bouldin: **1.5681**, Calinski-Harabasz: **590.88**).
- **Cluster Profiles**: In the optimal $k=2$ partitioning, Cluster 1 isolates anomalous products characterized by elevated co-review degrees ($\bar{d}_w = 837.28$), concentrated 5-star velocities, and a **90.82% ground-truth fake buyer prevalence** (versus 27.35% in Cluster 0).
- **Integrity**: Hardcoded clustering defaults have been eliminated from the backend; metrics are dynamically derived from saved sweep matrices.

---

## 12. SHAP Explainability & Multilingual Attribution
- **Global Decision Weight**: TreeSHAP attributions reveal that network structural features constitute **35.42%** of the model's total decision weight, led by `clustering_coef` and `w_degree`. Review dynamics account for **41.98%**, and image similarity features account for **22.61%**.
- **Multilingual Local Explanations**: The API provides real-time natural language explanations for every product prediction in three languages:
  - **English**: Structured attribution sentences.
  - **Hindi (हिंदी)**: Native Devanagari explanations.
  - **Hinglish**: Romanized colloquial Hindi popular among Indian consumers.

---

## 13. Risk Scoring & Trust-Adjusted Ratings
- **Continuous Fraud Risk Score (0–100)**:
  $$\text{Risk Score} = \text{round}\left(100 \times \left(0.85 \cdot p_{\text{calibrated}} + 0.15 \cdot \frac{\text{Rating Discrepancy}}{4.0}\right)\right)$$
- **Risk Tiers**:
  - `LOW` (Score < 30): Minimal indication of artificial review manipulation.
  - `MODERATE` (Score 30–64): Suspicious review velocity or elevated co-review clustering.
  - `HIGH` (Score $\ge$ 65): Strong statistical probability of coordinated fake review purchasing.
- **Trust-Adjusted Rating**: Bayesian shrinkage heuristic penalizing products with high fraud probability:
  $$R_{\text{adjusted}} = (1 - p_{\text{calibrated}}) \cdot R_{\text{observed}} + p_{\text{calibrated}} \cdot 3.50$$

---

## 14. FastAPI Production Backend
- **Port**: `http://localhost:8000` (Swagger UI at `/docs`)
- **Key Routes**:
  - `GET /health`: Service health and timestamp.
  - `POST /predict/product`: Schema-validated inference returning risk score, tier, and SHAP.
  - `GET /products/{id}`: Consumer product profile (ground-truth labels stripped).
  - `GET /products`: Paginated catalog.
  - `GET /models/registry`: Specifications for all 21 registered models.
  - `GET /models/metrics`: Research leaderboard benchmarks.
  - `GET /dashboard/overview`: Dynamic KPI metrics for the web landing view.
  - `GET /clustering/summary`: Dynamic K-Means sweep optimization summary.
  - `GET /network/summary`: Graph ML benchmark comparisons.
  - `GET /network/graph`: Interactive demonstration bipartite graph generator.
  - `GET /system/environment`: Live Python and library versions.
  - `GET /system/test-report`: Live verified automated test results.

---

## 15. React Frontend Architecture
The web client is built with React 18, TypeScript, and Tailwind CSS:
1. **Overview Dashboard** (`/dashboard`): Landing overview with real-time KPIs, class distributions, and feature gain charts.
2. **Product Checker** (`/product-checker`): Circular SVG risk gauge, manual feature overrides, catalog auto-fetch, and multilingual explanations.
3. **Model Comparison** (`/model-comparison`): Benchmark leaderboard with metric search, protocol filters, and interactive confusion matrices.
4. **Network Explorer** (`/network-explorer`): Interactive React Flow bipartite topology canvas with node inspector and noise controls.
5. **Clustering Dashboard** (`/clustering`): Sweep charts for Silhouette and Davies-Bouldin ($k=2..20$) and profile inspections.
6. **SHAP Explainability** (`/explainability`): Global feature ranking, modality weight pie chart, and correlation disclosures.
7. **Research Experiments** (`/experiments`): 98-experiment provenance catalog with audit metadata inspector.
8. **Reports & Exports** (`/reports`): Print-ready PDF report generator and CSV/JSON export utility.
9. **Settings & Compliance** (`/settings`): API latency monitor, 21 model schemas, and Bureau of Indian Standards (IS 19000:2022) regulatory guidelines.

---

## 16. Project Directory Structure
```
MAJOR-PROJECT/
├── src/                         # Python ML & Backend Core
│   ├── api/                     # FastAPI application (main.py)
│   ├── data/                    # Dataset loaders and validation schemas
│   ├── evaluation/              # Metrics, risk scoring, TreeSHAP explainer
│   ├── experiments/             # Experiment runners (Supervised, GNN, Optuna)
│   ├── features/                # Temporal, text, image, and network extractors
│   ├── graph/                   # GraphSAGE, Node2Vec, synthetic fraud rings
│   └── models/                  # Pipeline builders, classifiers, tuning
├── frontend/                    # React 18 + TypeScript Web Application
│   ├── src/                     # Components, pages, hooks, services, types
│   ├── src/test/                # Vitest unit and integration test suites
│   ├── package.json             # NPM package manifest
│   └── vite.config.ts           # Vite build configuration
├── tests/                       # Pytest Test Suite (37 automated tests)
│   ├── test_api.py              # HTTP endpoint validation tests
│   └── test_pipeline.py         # Leakage, GNN, Node2Vec, and calibration tests
├── outputs/                     # Empirical Artifacts
│   ├── models/                  # 21 serialized joblib model pipelines
│   ├── experiments/             # Benchmark CSVs, registries, and sweep metrics
│   ├── figures/                 # SHAP summary plots
│   ├── screenshots/             # 9 full-page end-to-end screenshots
│   └── test_report.json         # Automated verified test report artifact
├── docs/                        # Architecture, API contracts, model registry docs
├── scripts/                     # Verification and test generation scripts
├── data/                        # Reference structured product datasets
├── start_application.sh         # One-command full-stack launcher
├── requirements.txt             # Pinned Python dependencies
├── Dockerfile                   # Production container definition
├── docker-compose.yml           # Multi-container orchestration definition
└── README.md                    # Root project documentation
```

---

## 17. Installation Instructions

### Prerequisites
- Python 3.11 or 3.14
- Node.js v18+ and npm v9+

### Backend Setup
```bash
# Clone the repository
git clone https://github.com/Rishi119-mah/MAJOR-PROJECT.git
cd MAJOR-PROJECT

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Frontend Setup
```bash
cd frontend
npm install
cd ..
```

---

## 18. Backend Startup
```bash
source venv/bin/activate
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```
API endpoints will be live at `http://localhost:8000` with interactive Swagger docs at `http://localhost:8000/docs`.

---

## 19. Frontend Startup
```bash
cd frontend
npm run dev
```
The React analytics dashboard will be accessible at `http://localhost:5173`.

*(Alternatively, launch both services simultaneously using the unified launcher:)*
```bash
chmod +x start_application.sh
./start_application.sh
```

---

## 20. Running Tests

### Backend Test Suite (37 Tests, 100% Passing)
```bash
# Run pytest programmatically and generate verified test_report.json artifact:
python scripts/generate_test_report.py

# Or run pytest directly:
pytest tests/ -v
```

### Frontend Test Suite (22 Tests Across 5 Suites, 100% Passing)
```bash
cd frontend
npm test -- --run
```

### Production Build Verification
```bash
cd frontend
npm run build
```

### Model Registry Integrity Check (All 21 Models Passing)
```bash
PYTHONPATH=. python scripts/verify_model_registry.py
```

---

## 21. Model Artifacts and Dataset Setup
- **Trained Models**: All 21 serialized pipelines are tracked in `outputs/models/` (including calibrated XGBoost pipelines and CatBoost classifiers).
- **Core Dataset**: Bundled at `data/product_level_data_with_img_feats.csv.gz` (3,327 products, 28 features).
- **External Raw Datasets**: The 1 GB raw UCSD Amazon reviewer interaction archive is intentionally excluded from Git. For full raw graph reconstruction, see instructions in `docs/DATASET_SETUP.md`.

---

## 22. Screenshots
End-to-end full-page screenshots of all 9 application pages captured during live backend operation:

| Page | Preview |
| :--- | :--- |
| **01. Dashboard** | ![Dashboard](outputs/screenshots/01_dashboard.png) |
| **02. Product Checker** | ![Product Checker](outputs/screenshots/02_product_checker.png) |
| **03. Model Comparison** | ![Model Comparison](outputs/screenshots/03_model_comparison.png) |
| **04. Network Explorer** | ![Network Explorer](outputs/screenshots/04_network_explorer.png) |
| **05. Clustering** | ![Clustering](outputs/screenshots/05_clustering.png) |
| **06. TreeSHAP Explainability** | ![Explainability](outputs/screenshots/06_explainability.png) |
| **07. Research Experiments** | ![Experiments](outputs/screenshots/07_experiments.png) |
| **08. Reports & Exports** | ![Reports](outputs/screenshots/08_reports.png) |
| **09. Settings & Compliance** | ![Settings](outputs/screenshots/09_settings.png) |

---

## 23. Results and Empirical Evaluation

### Final Holdout Test Set Performance ($N=666$ Products)
| Architecture & Modality | Accuracy | Fake F1 | ROC-AUC | PR-AUC | Brier Score | ECE (10-bin) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM (Review + Network + Image)** | **0.8498** | **0.8193** | 0.9191 | 0.9024 | 0.1082 | 0.0481 |
| **XGBoost (Review + Network + Image)** | 0.8468 | 0.8179 | 0.9167 | 0.9015 | 0.1095 | 0.0494 |
| **Calibrated XGBoost (Review + Network + Image)** | 0.8468 | 0.8179 | 0.9167 | 0.9015 | **0.1012** | **0.0347** |
| **CatBoost (Review + Network + Image)** | 0.8358 | 0.8037 | **0.9238** | **0.9088** | 0.1141 | 0.0512 |
| **Stacking Classifier (Review + Network + Image)** | 0.8423 | 0.8112 | 0.9184 | 0.9031 | 0.1104 | 0.0478 |
| **XGBoost (Review + Network)** | 0.8343 | 0.8007 | 0.9148 | 0.8991 | 0.1189 | 0.0544 |
| **XGBoost (Review Only Baseline)** | 0.8078 | 0.7673 | 0.8912 | 0.8710 | 0.1342 | 0.0621 |

*(Key Finding: Adding network structure increases Fake F1 by **+3.34%** over review-only baselines. Adding image embeddings delivers a further **+1.72%** gain to **0.8193**).*

---

## 24. Limitations
1. **Raw Interaction Scale**: Generating dynamic bipartite graphs for millions of reviewers in real-time requires distributed graph infrastructure (e.g., Apache Spark GraphX or Neo4j).
2. **Review Text Sentiment**: Reviewer text TF-IDF is summarized at the product level; temporal sentiment shifts within individual product lifecycles are proxied via rating standard deviations.
3. **Cross-Platform Fraud Shift**: Fraud rings adapt across platforms (Amazon, Flipkart, Walmart); topological indicators may require periodic re-weighting.

---

## 25. Future Scope
1. **Heterogeneous Temporal Graph Neural Networks (HTGNN)**: Training continuous-time bipartite GNNs on raw timestamped interaction streams.
2. **LLM Agent-Assisted Review Text Auditing**: Integrating local open-source LLMs (e.g., Gemma 2) to perform syntactic collusion audits on suspicious text clusters.
3. **Browser Extension Deployment**: Packaging the React frontend client into a Chrome WebExtension for real-time fraud warnings while shopping online.

---

## License & Citation
This project is developed for final-year engineering degree requirements.  
Reference study: He, S., Hollenbeck, B., & Proserpio, D. (2022). *Detecting Fake Review Buyers Using Network Structure: Direct Evidence from Amazon.*
