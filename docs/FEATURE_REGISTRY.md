# FEATURE_REGISTRY.md — Feature Groups and Experiment Assignments

## Feature Group Registry

| Group ID | Group Name | Feature Count | Source Dataset | Notes |
|----------|-----------|---------------|----------------|-------|
| RV | Review Features | 12 | with_img, without_img | Available for all labelled products |
| NW | Network Features | 4 | with_img, without_img | Derived from product projection graph |
| IM | Image Similarity | 12 | with_img only | Available for 3,327 of 3,408 products |
| TX | TF-IDF Text | 1,000 | text_feats | Stemmed top-1000 unigram/bigram TF-IDF |
| IM_META | Image Metadata | 3 | with_img only | num_images, num_product_images, num_reviews |

## Experiment Feature Combinations

| Experiment ID | Feature Groups | Total Features | Dataset | Status |
|---------------|---------------|----------------|---------|--------|
| EXP_RV | RV | 12 | with_img | ✅ Original |
| EXP_NW | NW | 4 | with_img | ✅ Original |
| EXP_IM | IM | 12 | with_img | ✅ Original |
| EXP_NW_TOP2 | NW (top 2) | 2 | with_img | ✅ Original |
| EXP_RV_NW | RV + NW | 16 | with_img | ❌ MISSING in original |
| EXP_RV_NW_IM | RV + NW + IM | 28 | with_img | ✅ Original ("All Features") |
| EXP_TX | TX | 1,000 | text_feats | ✅ Original |
| EXP_RV_NW_TX | RV + NW + TX | 1,016 | merged | ⏳ New |
| EXP_ALL | RV + NW + IM + TX | 1,028 | merged | ⏳ New |

## Feature Lists

### RV: Review Features
```
tfidf_review_body, n_of_reviews, avg_review_rating,
avg_days_between_reviews, stdev_days_between_reviews,
max_days_between_reviews, min_days_between_reviews,
share_helpful_reviews, share_1star, share_5star,
share_photo, std_review_len
```

### NW: Network Features
```
pagerank, w_degree, clustering_coef, eigenvector_cent
```

### IM: Image Similarity Features
```
min_sim, max_sim, mean_sim, std_sim,
min_sim_review, max_sim_review, mean_sim_review, std_sim_review,
min_sim_product, max_sim_product, mean_sim_product, std_sim_product
```

### NW_TOP2: Top-2 Network Features
```
eigenvector_cent, clustering_coef
```

