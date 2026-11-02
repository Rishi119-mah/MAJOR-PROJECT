# DATA_DICTIONARY.md — Feature Definitions and Data Sources

## Datasets Overview

### Labelled Product-Level Data

**Source:** Authors' proprietary dataset of Amazon products with verified fake-review-buyer labels.

| Dataset Variant | Products | Features | Purpose |
|-----------------|----------|----------|---------|
| with_img_feats | 3,327 | 33 | Full feature set including image similarity |
| without_img_feats | 3,408 | 18 | Base features; 81 more products than img variant |
| text_feats | 3,327 | 1,002 | Top-1000 TF-IDF features; same products as img |

### Unlabelled UCSD Product-Level Data

| Dataset | Products | Features | Purpose |
|---------|----------|----------|---------|
| UCSD_product_level_data | 64,585 | 21 | Unsupervised clustering / semi-supervised prediction |

### Review-Level Data (NOT AVAILABLE LOCALLY)

| Dataset | Expected Size | Purpose |
|---------|---------------|---------|
| UCSD_home_and_kitchen_reviews | ~1 GB | Bipartite graph construction for clustering |

---

## Target Variable

| Column | Type | Values | Description |
|--------|------|--------|-------------|
| `fake` | int64 | 0, 1 | **1** = product buys fake reviews, **0** = genuine. Positive class is **fake=1**. |

### Distribution

| Dataset | Genuine (0) | Fake (1) | Total | Fake % |
|---------|-------------|----------|-------|--------|
| with_img | 1,927 | 1,400 | 3,327 | 42.08% |
| without_img | 1,936 | 1,472 | 3,408 | 43.19% |

---

## Feature Definitions

### Review Features (12)

| Feature | Type | Description | Source |
|---------|------|-------------|--------|
| `tfidf_review_body` | float64 | Mean TF-IDF score of review text bodies | Aggregated from reviews |
| `n_of_reviews` | int64 | Total number of reviews for the product | Review count |
| `avg_review_rating` | float64 | Average star rating across all reviews | Review ratings |
| `avg_days_between_reviews` | float64 | Mean interval (days) between consecutive reviews | Review timestamps |
| `stdev_days_between_reviews` | float64 | Standard deviation of inter-review intervals | Review timestamps |
| `max_days_between_reviews` | float64 | Maximum gap between consecutive reviews | Review timestamps |
| `min_days_between_reviews` | float64 | Minimum gap between consecutive reviews | Review timestamps |
| `share_helpful_reviews` | float64 | Proportion of reviews marked as helpful | Review helpfulness votes |
| `share_1star` | float64 | Proportion of 1-star reviews | Review ratings |
| `share_5star` | float64 | Proportion of 5-star reviews | Review ratings |
| `share_photo` | float64 | Proportion of reviews containing photos | Review content |
| `std_review_len` | float64 | Standard deviation of review text lengths | Review text |

### Network Features (4)

| Feature | Type | Description | Graph Type |
|---------|------|-------------|------------|
| `pagerank` | float64 | PageRank centrality in weighted product projection graph | Product projection |
| `w_degree` | int64 | Weighted degree in product projection (sum of shared-reviewer counts) | Product projection |
| `clustering_coef` | float64 | Local clustering coefficient in product graph | Product projection |
| `eigenvector_cent` | float64 | Eigenvector centrality in product graph | Product projection |

### Image Similarity Features (12)

| Feature | Type | Description |
|---------|------|-------------|
| `min_sim` | float64 | Minimum pairwise image similarity (all images) |
| `max_sim` | float64 | Maximum pairwise image similarity (all images) |
| `mean_sim` | float64 | Mean pairwise image similarity (all images) |
| `std_sim` | float64 | Std dev of pairwise image similarity (all images) |
| `min_sim_review` | float64 | Minimum similarity among review images |
| `max_sim_review` | float64 | Maximum similarity among review images |
| `mean_sim_review` | float64 | Mean similarity among review images |
| `std_sim_review` | float64 | Std dev of similarity among review images |
| `min_sim_product` | float64 | Minimum similarity among product listing images |
| `max_sim_product` | float64 | Maximum similarity among product listing images |
| `mean_sim_product` | float64 | Mean similarity among product listing images |
| `std_sim_product` | float64 | Std dev of similarity among product listing images |

### Image Metadata Features (3, in img dataset only)

| Feature | Type | Description |
|---------|------|-------------|
| `num_images` | float64 | Total images associated with product |
| `num_product_images` | float64 | Number of product listing images |
| `num_reviews` | float64 | Number of reviews with images |

### Text Features (1,000)

| Feature Group | Count | Description |
|---------------|-------|-------------|
| TF-IDF unigrams/bigrams | 1,000 | Stemmed top-1000 TF-IDF features from review text. Column names are stemmed tokens (e.g., 'abil', 'absolut love', 'bluetooth'). |

### UCSD Extra Network Features (4, in UCSD dataset only)

| Feature | Type | Description |
|---------|------|-------------|
| `degree` | int64 | Unweighted degree centrality |
| `w_pagerank` | float64 | Weighted PageRank |
| `w_clustering_coef` | float64 | Weighted clustering coefficient |
| `w_eigenvector_cent` | float64 | Weighted eigenvector centrality |

---

## Identifiers

| Column | Type | Description |
|--------|------|-------------|
| `product_ID` | int64 | Unique numeric product identifier (anonymized). Present in ALL datasets. |
| `reviewer_ID` | (varies) | Reviewer identifier. Present only in review-level data. |

---

## Network Construction

The product projection graph is constructed as follows:

1. **Bipartite graph** B = (Reviewers, Products, Edges)
   - A reviewer node connects to a product node if the reviewer reviewed the product
2. **Product projection** P = weighted_projected_graph(B, products)
   - Two products are connected if they share at least one reviewer
   - Edge weight = number of shared reviewers (simple count)
3. **Centrality features** computed on P:
   - PageRank (α=0.85)
   - Eigenvector centrality (max_iter=500)
   - Clustering coefficient
   - Weighted degree

> ⚠️ **Known risk:** If reviewer_ID values overlap with product_ID values, nodes may collide in the bipartite graph. The original code does NOT use typed node prefixes.

