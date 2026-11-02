"""
Central configuration for the Fake Review Detection project.

All paths, feature definitions, model hyperparameters, and experiment
settings are defined here so every other module imports a single source
of truth.
"""

from pathlib import Path

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
EXPERIMENT_DIR = OUTPUT_DIR / "experiments"
FIGURE_DIR = OUTPUT_DIR / "figures"

# Ensure output directories exist
for _d in (OUTPUT_DIR, MODEL_DIR, EXPERIMENT_DIR, FIGURE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# Dataset filenames
# ──────────────────────────────────────────────
DATA_WITH_IMG = DATA_DIR / "product_level_data_with_img_feats.csv.gz"
DATA_WITHOUT_IMG = DATA_DIR / "product_level_data_without_img_feats.csv.gz"
DATA_TEXT = DATA_DIR / "product_level_data_text_feats.csv.gz"
DATA_UCSD = DATA_DIR / "UCSD_product_level_data.csv.gz"

# ──────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────
RANDOM_SEED = 42
TEST_SIZE = 0.20
CV_SPLITS = 5
CV_REPEATS = 2

# ──────────────────────────────────────────────
# Target column
# ──────────────────────────────────────────────
TARGET = "fake"
ID_COL = "product_ID"

# ──────────────────────────────────────────────
# Feature groups  (match original code exactly)
# ──────────────────────────────────────────────
REVIEW_FEATURES: list[str] = [
    "tfidf_review_body",
    "n_of_reviews",
    "avg_review_rating",
    "avg_days_between_reviews",
    "stdev_days_between_reviews",
    "max_days_between_reviews",
    "min_days_between_reviews",
    "share_helpful_reviews",
    "share_1star",
    "share_5star",
    "share_photo",
    "std_review_len",
]

NETWORK_FEATURES: list[str] = [
    "pagerank",
    "w_degree",
    "clustering_coef",
    "eigenvector_cent",
]

IMAGE_FEATURES: list[str] = [
    "min_sim",
    "max_sim",
    "mean_sim",
    "std_sim",
    "min_sim_review",
    "max_sim_review",
    "mean_sim_review",
    "std_sim_review",
    "min_sim_product",
    "max_sim_product",
    "mean_sim_product",
    "std_sim_product",
]

IMAGE_META_FEATURES: list[str] = [
    "num_images",
    "num_product_images",
    "num_reviews",
]

# Convenience combinations
ALL_STRUCTURED_FEATURES: list[str] = (
    REVIEW_FEATURES + NETWORK_FEATURES + IMAGE_FEATURES
)

REVIEW_NETWORK_FEATURES: list[str] = REVIEW_FEATURES + NETWORK_FEATURES

# ──────────────────────────────────────────────
# Experiment configurations
# ──────────────────────────────────────────────
FEATURE_COMBOS: dict[str, list[str]] = {
    "review_only": REVIEW_FEATURES,
    "review_network": REVIEW_NETWORK_FEATURES,
    "review_network_image": ALL_STRUCTURED_FEATURES,
    # "text_only" is handled separately (uses text df)
}

# Models to run in improved experiments
IMPROVED_MODELS: list[str] = [
    "dummy",
    "logistic_regression",
    "random_forest",
    "svm_linear",
    "svm_rbf",
    "xgboost",
    # "lightgbm",   # Uncomment when lightgbm is installable
    # "catboost",   # Uncomment when catboost is installable
]

# Legacy models (exact original params, for reproduction)
LEGACY_MODELS: list[str] = [
    "logistic_regression",
    "random_forest",
    "svm_linear",
    "xgboost",
]

LEGACY_FEATURE_COMBOS: dict[str, list[str]] = {
    "review_only": REVIEW_FEATURES,
    "image_only": IMAGE_FEATURES,
    "network_only": NETWORK_FEATURES,
    "top2_network": ["eigenvector_cent", "clustering_coef"],
    "all_structured": ALL_STRUCTURED_FEATURES,
    # "text_only" handled separately
}
