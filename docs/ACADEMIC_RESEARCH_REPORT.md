# Multimodal Fake Review Buyer Detection via Network Centrality, Gradient Boosting, and Explainable AI
**A Reproducible Empirical Study on Amazon Product-Review Networks**

**Authors:** Rishi Maheshwari et al.  
**Affiliation:** Final Year Major Project Research Report  
**Date:** November 2026  
**Repository:** https://github.com/dadepro/fake_review_detection  

---

## Abstract
Online consumer reviews represent a multi-billion-dollar driver of e-commerce purchasing decisions. Consequently, fraudulent sellers systematically purchase deceptive, positive reviews to manipulate platform search algorithms and artificial star ratings. Prior text-only natural language processing approaches exhibit vulnerability to modern linguistic mimicry. In this work, building upon the foundational empirical insights of He et al. (2022), we present a comprehensive, multi-modal, explainable machine learning system that combines review text statistics, product co-review network topology, image similarity, and temporal dynamics to identify products that purchase fake reviews on Amazon. 

We systematically audit the original research reproduction codebase, eliminate data leakage risks by establishing stratified cross-validation protocols, and benchmark 9 supervised algorithms across 6 feature sets. Gradient boosting models (XGBoost, LightGBM, CatBoost) achieve superior predictive performance, with our tuned XGBoost achieving **0.9199 ROC-AUC**, **0.9069 PR-AUC**, and **0.8179 Fake-Class F1-Score** on holdout test sets, and **0.9386 ROC-AUC / 0.8430 F1** on baseline configurations. Using TreeSHAP explainability, we discover that network features collectively account for **35.42%** of total model importance—with clustering coefficient (16.07%) and eigenvector centrality (15.38%) driving the classification. Furthermore, we construct an optimal K-Means cluster analysis ($k=2..20$), validate synthetic fraudulent review-ring robustness achieving 100% detection recall, formulate a consumer-facing 0–100 Product Risk Indicator with a trust-adjusted rating formula, and deploy a production-grade FastAPI microservice supporting multilingual explanations in English, Hindi, and Hinglish under Indian regulatory frameworks (BIS IS 19000:2022 and CCPA Guidelines 2022).

---

## 1. Introduction & Research Problem
Amazon and global e-commerce marketplaces suffer from coordinated underground review-buying rings. Fraudulent sellers hire third-party brokers (often coordinated over encrypted messaging channels) who recruit consumers to purchase specific products and leave verified 5-star reviews with attached photos in exchange for full refunds and commissions.

Because these reviews are submitted by human buyers who purchase real units, traditional single-review text classifiers fail to reliably detect them. The pivotal insight established by He et al. (2022) is that **colluding products inevitably share common reviewers**, forming dense, anomalous bipartite graph structures in the reviewer-product interaction network.

### Objectives of this Project:
1. **Audited Reproduction:** Verify the original 2022 reproduction code and dataset integrity without data leakage.
2. **Multi-Algorithm Benchmark:** Evaluate 9 classifiers (Dummy, Logistic Regression, Random Forest, SVM Linear, SVM RBF, XGBoost, LightGBM, CatBoost, Stacking Ensemble) across structured feature combinations.
3. **Rigorous Feature Impact Assessment:** Quantify the incremental benefit of Network features over Review-only features.
4. **Empirical Explainability:** Verify feature importance via TreeSHAP and test unverified claims regarding individual feature weights.
5. **Graph Machine Learning Benchmark:** Empirically evaluate handcrafted graph metrics against learned embeddings (Spectral/Node2Vec) and inductive GraphSAGE neighborhood aggregation.
6. **Production Microservice:** Deploy a high-throughput, tested FastAPI backend with localization and regulatory alignment for Indian consumer protection.

---

## 2. Dataset Inventory & Schema Integrity

We audited all four datasets locally available in the project workspace:

| Dataset File | Rows | Features | Has Ground Truth `fake` Label? | Positive Class Rate | Cleanliness Status |
|---|---|---|---|---|---|
| `product_level_data_with_img_feats.csv.gz` | 3,327 | 33 | Yes | 42.08% (1,400 fake / 1,927 genuine) | 0 missing, 0 inf, 0 duplicates |
| `product_level_data_text_feats.csv.gz` | 3,327 | 1,002 | Yes | 42.08% (1,400 fake / 1,927 genuine) | 1,000 stemmed TF-IDF tokens |
| `product_level_data_without_img_feats.csv.gz` | 3,408 | 18 | Yes | 43.19% (1,472 fake / 1,936 genuine) | Base review + network features |
| `UCSD_product_level_data.csv.gz` | 64,585 | 21 | No (Unlabelled) | N/A | Home & Kitchen category products |

*Note on Missing Review File:* `UCSD_home_and_kitchen_reviews.csv.gz` (~1GB) is an external Dropbox dependency needed for raw reviewer bipartite reconstruction on the full UCSD set. Our audit identified this dependency and safely decoupled the core machine learning and graph validation pipelines.

---

## 3. Supervised Classification Results

We evaluated all models under both:
1. **Legacy Mode:** Exact original parameters (no stratification, StandardScaler on all features, single 80/20 split).
2. **Improved Stratified Mode:** Stratified train/test split, model-specific preprocessing (scaling for linear/SVM models; unscaled for tree models), and comprehensive metric tracking.

### 3.1 Feature Progression Impact (XGBoost Classifier)

| Feature Group | Features | Accuracy | Fake F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| **Review Only** | 12 | 80.33% | 0.7673 | 0.8774 | 0.8463 |
| **Network Only** | 4 | 80.33% | 0.7698 | 0.8596 | 0.8259 |
| **Top-2 Network (Eigen+CC)** | 2 | 79.58% | 0.7606 | 0.8707 | 0.8353 |
| **Review + Network** | 16 | **83.18%** | **0.8007** | **0.9137** | **0.8961** |
| **Review + Network + Image** | 28 | **84.68%** | **0.8179** | **0.9199** | **0.9069** |
| **All Text (TF-IDF)** | 1,000 | 76.88% | 0.6992 | 0.8484 | 0.8194 |

> **Key Finding:** Adding network features to review features increases ROC-AUC by **+3.63%** (from 0.8774 to 0.9137) and Fake F1 by **+3.34%** (0.7673 to 0.8007). Adding image similarity features provides an additional **+0.62%** AUC gain. Review text alone (TF-IDF) performs significantly worse (F1 = 0.6992).

### 3.2 5x2 Repeated Stratified Cross-Validation Summary (Review + Network + Image)

| Model | ROC-AUC (Mean ± Std) | PR-AUC (Mean ± Std) | Accuracy (Mean ± Std) | Fake F1 (Mean ± Std) |
|---|---|---|---|---|
| **CatBoost** | **0.9325 ± 0.0103** | 0.9182 ± 0.0124 | **86.05% ± 1.47%** | **0.8324 ± 0.0187** |
| **XGBoost** | 0.9300 ± 0.0118 | 0.9165 ± 0.0131 | 85.72% ± 1.29% | 0.8281 ± 0.0161 |
| **LightGBM** | 0.9295 ± 0.0115 | 0.9150 ± 0.0130 | 85.66% ± 1.28% | 0.8270 ± 0.0162 |
| **Random Forest** | 0.9248 ± 0.0118 | 0.9079 ± 0.0138 | 84.73% ± 1.34% | 0.8166 ± 0.0171 |
| **SVM (RBF Kernel)** | 0.9191 ± 0.0126 | 0.8994 ± 0.0145 | 84.10% ± 2.23% | 0.8029 ± 0.0279 |
| **Logistic Regression**| 0.9142 ± 0.0121 | 0.8920 ± 0.0152 | 84.34% ± 1.45% | 0.8087 ± 0.0192 |

---

## 4. Empirical SHAP Feature Importance Analysis

Using TreeSHAP on our trained XGBoost pipeline, we computed exact global feature importance (mean absolute SHAP value and percentage contribution).

### Top 10 Most Influential Features:
1. `clustering_coef` (Network): **16.07%** of total importance
2. `eigenvector_cent` (Network): **15.38%** of total importance
3. `n_of_reviews` (Review): **9.88%**
4. `share_photo` (Review): **9.04%**
5. `max_days_between_reviews` (Review/Temporal): **5.69%**
6. `share_5star` (Review): **5.19%**
7. `min_sim` (Image): **3.96%**
8. `avg_days_between_reviews` (Review/Temporal): **3.76%**
9. `min_sim_product` (Image): **3.65%**
10. `w_degree` (Network): **2.88%**

### Investigation of Presentation Claim (~21% w_degree):
- The presentation hypothesized that `w_degree` represented ~21% of total importance.
- Our empirical SHAP calculation reveals that **Total Network Importance is 35.42%**, but the dominant contributors are `clustering_coef` (16.07%) and `eigenvector_cent` (15.38%), with `w_degree` at 2.88%.
- Together, the top-2 network features represent **31.45%** of all decision weight, verifying the authors' decision in `classification.py` to isolate the `['eigenvector_cent', 'clustering_coef']` pair.

---

## 5. Graph Machine Learning & Embedding Benchmarks

We implemented and compared handcrafted graph centrality metrics with learned spectral/Node2Vec graph embeddings and inductive GraphSAGE neighborhood aggregation:

| Method | Fake-Class F1 | ROC-AUC | Accuracy |
|---|---|---|---|
| **Handcrafted Network Centrality + XGB** | **0.8035** | **0.9193** | **83.58%** |
| **Spectral Graph Embeddings (d=16) + XGB** | 0.7864 | 0.8992 | 82.55% |
| **GraphSAGE Neighborhood Aggregations + XGB** | 0.7708 | 0.8789 | 80.65% |
| *PPT Stated Claim (GraphSAGE)* | *0.8220 (Unverified in original repo)* | *N/A* | *N/A* |

> **Conclusion:** Handcrafted network centrality metrics (PageRank, eigenvector centrality, clustering coefficient, weighted degree) outperform raw graph embeddings because domain-specific topological bottlenecks directly map to reviewer coordination.

---

## 6. Unsupervised Clustering & Anomaly Detection

1. **K-Means Sweep ($k=2..20$):**
   - Optimal silhouette score occurs at $k=2$ (Silhouette = 0.2474).
   - At $k=20$ (paper default), silhouette score is 0.1334.
2. **Cluster Separation Analysis:**
   - In $k=20$, specific clusters exhibit severe concentrations of fake review buyers:
     - **Cluster 0:** 90.82% ground truth fake rate (our model predicted 88.95%).
     - **Cluster 2:** 77.12% ground truth fake rate (our model predicted 76.84%).
     - **Cluster 1:** 27.35% ground truth fake rate (our model predicted 24.86%).
3. **Synthetic Fraud-Ring Robustness:**
   - Coordinated review-ring simulation across 9 noise and ring size scenarios demonstrated **100% detection recall** based on weighted degree and clustering coefficient shifts.

---

## 7. Product Risk Scoring & Trust-Adjusted Rating

To provide actionable value to consumers, we established:
1. **Product Risk Score (0–100):** Continuous calibrated score where:
   - 0–30: LOW_RISK
   - 31–65: MODERATE_RISK
   - 66–100: HIGH_RISK
2. **Trust-Adjusted Rating Formula:**
   $$\text{Adjusted Rating} = (1 - p_{\text{fake}}) \times \text{Displayed Rating} + p_{\text{fake}} \times \text{Reference Rating}$$
   where $\text{Reference Rating} = 3.50$ stars (empirical category baseline). A 4.8-star product with $p_{\text{fake}} = 0.85$ is adjusted down to **3.70 stars**.

---

## 8. Regulatory Context & Compliance (India)

The system is aligned with Indian statutory standards:
- **BIS IS 19000:2022:** Prescribes guidelines for online review verification, traceability, and anti-manipulation moderation.
- **CCPA Guidelines (2022):** Section 21 of the Consumer Protection Act, 2019 imposes penalties up to ₹10–50 Lakhs for deceptive endorsements and fake online reviews.

---

## 9. Conclusion
This study provides the first complete, leakage-safe, explainable reproduction and production deployment of the Amazon fake-review-buyer detection framework. By demonstrating that network centrality features account for over 35% of model decisions and drive ROC-AUC past 0.93, we confirm that network structure is essential for safeguarding modern e-commerce marketplaces against review manipulation.

