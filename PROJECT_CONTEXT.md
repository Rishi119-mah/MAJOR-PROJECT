# PROJECT_CONTEXT.md — Fake Review Detection System
> Last Updated: 2026-11-02  
> Purpose: Allow any future development session to continue without repeating the audit.

## 1. Project Identity

- **Title:** Detecting Fake Review Buyers Using Machine Learning and Network Analysis
- **Reference Paper:** He, S., Hollenbeck, B., Overgoor, G., Proserpio, D., & Tosyali, A. (2022). "Detecting Fake Review Buyers Using Network Structure: Direct Evidence from Amazon"
- **Repository Origin:** https://github.com/dadepro/fake_review_detection
- **Local Workspace:** `/Users/rishimaheshwari/Downloads/MAJOR PROJECT FINAL`

## 2. Environment

| Item | Value |
|------|-------|
| Python | 3.14.6 (via Homebrew) |
| Virtual Env | `./venv/` |
| scikit-learn | 1.9.1 |
| XGBoost | 3.4.1 |
| pandas | 3.1.0 |
| networkx | 3.7 |
| scipy | 1.18.1 |
| matplotlib | 3.11.2 |

## 3. Dataset Inventory

### Available Locally (4 files)

| File | Rows | Cols | Labels? | Pos% | Notes |
|------|------|------|---------|------|-------|
| `product_level_data_with_img_feats.csv.gz` | 3,327 | 33 | ✅ fake | 42.08% | Review + Network + Image features. Subset of noimg (81 fewer products — those lacking images). |
| `product_level_data_text_feats.csv.gz` | 3,327 | 1,002 | ✅ fake | 42.08% | Top-1000 TF-IDF stemmed unigram/bigram features. Same 3,327 products as img dataset. |
| `product_level_data_without_img_feats.csv.gz` | 3,408 | 18 | ✅ fake | 43.19% | Review + Network features. Superset of img (81 extra products). Label distribution: 1,936 genuine / 1,472 fake. |
| `UCSD_product_level_data.csv.gz` | 64,585 | 21 | ❌ | N/A | Unlabelled UCSD Home & Kitchen. Has both weighted and unweighted graph features. |

### NOT Available Locally (1 file)

| File | Expected Size | Source |
|------|---------------|--------|
| `UCSD_home_and_kitchen_reviews.csv.gz` | ~1 GB | [Dropbox link](https://www.dropbox.com/s/o2jv9uw7emd0dgy/UCSD_home_and_kitchen_reviews.csv.gz?dl=0) |

> This file is required by `clustering.py` for bipartite graph construction. Clustering experiments are BLOCKED until downloaded.

### Key Dataset Relationships

- `with_img_feats` (3,327) ⊂ `without_img_feats` (3,408) — 81 products lack image data
- `text_feats` (3,327) = same products as `with_img_feats`
- All 3,327/3,408 labelled products also appear in UCSD (64,585)
- Labels are consistent across datasets for shared products (verified)
- Feature values are identical for shared features between img/noimg datasets (verified)
- UCSD has NO fake labels — used only for unsupervised clustering / semi-supervised inference

### Feature Groups

| Group | Count | Columns |
|-------|-------|---------|
| Review | 12 | tfidf_review_body, n_of_reviews, avg_review_rating, avg/stdev/max/min_days_between_reviews, share_helpful_reviews, share_1star, share_5star, share_photo, std_review_len |
| Network | 4 | pagerank, w_degree, clustering_coef, eigenvector_cent |
| Image Similarity | 12 | min/max/mean/std_sim, min/max/mean/std_sim_review, min/max/mean/std_sim_product |
| Image Meta | 3 | num_images, num_product_images, num_reviews (in img dataset only) |
| TF-IDF Text | 1,000 | Stemmed unigrams/bigrams (separate dataset) |
| UCSD Extra Network | 4 | degree, w_pagerank, w_clustering_coef, w_eigenvector_cent |

## 4. Original Script Inventory

### `code/classification.py` (128 lines)
- Loads `with_img_feats` and `text_feats` from CWD
- 4 models: LogisticRegression(max_iter=400), RandomForest(n_estimators=100, max_depth=40), SVC(kernel='linear'), XGBClassifier(defaults)
- Single 80/20 split, random_state=42, NO stratification
- StandardScaler on all features (including trees — unnecessary but harmless for trees)
- Reports: AUC, Accuracy, TN-rate, TP-rate, weighted F1
- 6 experiments: Review, Image, Network, Top-2-Network, All(Review+Image+Network), TF-IDF Text
- MISSING: Review+Network experiment, cross-validation, PR-AUC, precision, confusion matrix export

### `code/clustering.py` (166 lines)
- Requires `UCSD_home_and_kitchen_reviews.csv.gz` (NOT available locally)
- Loads all 3 UCSD-related files
- Builds bipartite graph using `iterrows` (slow for large datasets)
- Custom `weighted_projected_graph` function
- K-Means k=20 on UCSD review+network features
- Per-cluster: rebuilds bipartite graph → projection → network features → RF prediction
- RF has n_estimators=1200 (different from classification.py's 100!)
- `p_fake` is RF predicted probability, NOT ground-truth label
- Pivot table counts p_fake >= 0.5 as "fake"

## 5. Existing Code Structure

```
MAJOR PROJECT FINAL/
├── README.md
├── requirements.txt
├── code/
│   ├── classification.py   (original, untouched)
│   └── clustering.py        (original, untouched)
├── data/
│   ├── product_level_data_with_img_feats.csv.gz
│   ├── product_level_data_text_feats.csv.gz
│   ├── product_level_data_without_img_feats.csv.gz
│   └── UCSD_product_level_data.csv.gz
├── venv/                    (Python 3.14 virtual environment)
├── docs/                    (new — project documentation)
├── src/                     (new — modular source code)
├── legacy/                  (new — preserved original code)
├── tests/                   (new — automated tests)
├── outputs/                 (new — experiments, models, figures)
└── configs/                 (new — experiment configurations)
```

## 6. PPT File Status

- **NOT found** on local system
- Referenced as `Fake_Review_Detection_Review1_Final.pptx`
- Cannot verify PPT claims (GraphSAGE F1=0.822, SHAP 21% weighted degree, etc.)
- All such claims are treated as UNVERIFIED until reproduced

## 7. Critical Issues Identified

See `REPOSITORY_AUDIT.md` for detailed findings.

### Blocking Issues
1. `UCSD_home_and_kitchen_reviews.csv.gz` missing — clustering pipeline blocked
2. No PPT file available for claim verification

### Code Quality Issues  
3. No stratification in train/test split
4. Linear SVM instead of RBF SVM
5. No cross-validation
6. No PR-AUC or precision reporting
7. Missing Review+Network feature experiment
8. `iterrows` in graph construction (performance)
9. No model persistence
10. No error handling
11. No reproducibility framework

## 8. Implementation Phases Status

| Phase | Status | Notes |
|-------|--------|-------|
| 0: Repository Audit | ✅ COMPLETE | This document |
| 1: Data & Preprocessing | 🔄 IN PROGRESS | |
| 2: Supervised ML Pipeline | 🔄 IN PROGRESS | |
| 3: Hyperparameter Tuning | ⏳ PENDING | |
| 4: Network Analysis | ⏳ PENDING | Blocked by missing review data |
| 5: Node2Vec & GNNs | ⏳ PENDING | |
| 6: Temporal Features | ⏳ PENDING | |
| 7: NLP & Image | ⏳ PENDING | |
| 8: SHAP | ⏳ PENDING | |
| 9: Clustering | ⏳ PENDING | Partially blocked |
| 10: Synthetic Robustness | ⏳ PENDING | |
| 11: Risk Score | ⏳ PENDING | |
| 12: Dashboard Backend | ⏳ PENDING | |
| 13: FastAPI | ⏳ PENDING | |
| 14: Hindi/Hinglish | ⏳ PENDING | |
| 15: Testing | ⏳ PENDING | |
| 16: Docker | ⏳ PENDING | |
| 17: Report & PPT | ⏳ PENDING | |

