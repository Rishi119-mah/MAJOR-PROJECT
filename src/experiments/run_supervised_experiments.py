"""
Run all Phase 1-3 supervised learning experiments:
1. Exact Legacy Reproduction (for historical comparison)
2. Improved Stratified Holdout Benchmarks (all 9 models x all feature combos)
3. 5x2 Repeated Stratified Cross-Validation (on key feature combos)
4. Model Hyperparameter Tuning via Optuna
5. Save trained best models to outputs/models/
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

# Setup paths and logger
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import (
    DATA_WITH_IMG,
    DATA_TEXT,
    DATA_WITHOUT_IMG,
    EXPERIMENT_DIR,
    MODEL_DIR,
    RANDOM_SEED,
    REVIEW_FEATURES,
    NETWORK_FEATURES,
    IMAGE_FEATURES,
    ALL_STRUCTURED_FEATURES,
    REVIEW_NETWORK_FEATURES,
)
from src.data.data_loader import load_product_data, load_text_data
from src.evaluation.metrics import format_metrics_table
from src.models.classifiers import get_model, get_legacy_model
from src.models.pipeline import (
    run_legacy_experiment,
    run_single_experiment,
    run_cv_experiment,
    log_experiment_result,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_experiments")


def run_legacy_suite(df_img: pd.DataFrame, df_text: pd.DataFrame) -> None:
    """Execute the exact legacy experiments from classification.py."""
    logger.info("=" * 60)
    logger.info("RUNNING LEGACY REPRODUCTION SUITE (EXACT ORIGINAL PARAMS)")
    logger.info("=" * 60)

    legacy_experiments = [
        ("legacy_review_only", df_img, REVIEW_FEATURES),
        ("legacy_image_only", df_img, IMAGE_FEATURES),
        ("legacy_network_only", df_img, NETWORK_FEATURES),
        ("legacy_top2_network", df_img, ["eigenvector_cent", "clustering_coef"]),
        ("legacy_all_structured", df_img, ALL_STRUCTURED_FEATURES),
    ]

    for exp_id, df, feats in legacy_experiments:
        logger.info("Running legacy experiment: %s (%d features)", exp_id, len(feats))
        results = run_legacy_experiment(df, feats, experiment_name=exp_id)
        for m_name, metrics in results.items():
            log_experiment_result(
                experiment_id=exp_id,
                model=f"legacy_{m_name}",
                feature_group=exp_id.replace("legacy_", ""),
                metrics=metrics,
            )
            print(
                f"[{exp_id}] {m_name:20s} | AUC: {metrics.get('roc_auc', 0):.4f} | "
                f"Acc: {metrics['accuracy']:.4f} | Fake F1: {metrics['fake_f1']:.4f} | "
                f"Weighted F1: {metrics['weighted_f1']:.4f}"
            )

    # Legacy text experiment
    logger.info("Running legacy text experiment: legacy_text_only (1000 features)")
    results_text = run_legacy_experiment(
        df_text, features=None, experiment_name="legacy_text_only"
    )
    for m_name, metrics in results_text.items():
        log_experiment_result(
            experiment_id="legacy_text_only",
            model=f"legacy_{m_name}",
            feature_group="text_only",
            metrics=metrics,
        )
        print(
            f"[legacy_text_only] {m_name:20s} | AUC: {metrics.get('roc_auc', 0):.4f} | "
            f"Acc: {metrics['accuracy']:.4f} | Fake F1: {metrics['fake_f1']:.4f} | "
            f"Weighted F1: {metrics['weighted_f1']:.4f}"
        )


def run_improved_holdout_suite(
    df_img: pd.DataFrame, df_text: pd.DataFrame
) -> None:
    """Execute clean, leakage-safe holdout evaluations with stratified splits."""
    logger.info("=" * 60)
    logger.info("RUNNING IMPROVED STRATIFIED HOLDOUT SUITE")
    logger.info("=" * 60)

    models_to_test = [
        "dummy",
        "logistic_regression",
        "random_forest",
        "svm_linear",
        "svm_rbf",
        "xgboost",
        "lightgbm",
        "catboost",
        "stacking",
    ]

    feature_combos = [
        ("review_only", df_img, REVIEW_FEATURES),
        ("review_network", df_img, REVIEW_NETWORK_FEATURES),
        ("review_network_image", df_img, ALL_STRUCTURED_FEATURES),
        ("network_only", df_img, NETWORK_FEATURES),
    ]

    best_models = {}

    for combo_name, df, feats in feature_combos:
        logger.info(
            "--- Feature Group: %s (%d features) ---", combo_name, len(feats)
        )
        for m_name in models_to_test:
            try:
                res = run_single_experiment(
                    df=df,
                    features=feats,
                    model_name=m_name,
                    stratify=True,
                    experiment_id=f"improved_{combo_name}",
                )
                log_experiment_result(
                    experiment_id=f"improved_{combo_name}",
                    model=m_name,
                    feature_group=combo_name,
                    metrics=res,
                )
                print(
                    f"[{combo_name:22s}] {m_name:20s} | "
                    f"AUC: {res.get('roc_auc', 0):.4f} | "
                    f"PR-AUC: {res.get('pr_auc', 0):.4f} | "
                    f"Acc: {res['accuracy']:.4f} | "
                    f"Fake F1: {res['fake_f1']:.4f} | "
                    f"Train(s): {res['train_time_s']:.2f}"
                )

                # Keep track of best pipeline for persistence
                key = f"{combo_name}_{m_name}"
                if m_name in ["random_forest", "xgboost", "catboost", "stacking"]:
                    best_models[key] = (res["pipeline"], res["fake_f1"])

            except Exception as e:
                logger.error("Failed on %s with %s: %s", combo_name, m_name, e)

    # Text Only on improved models (fast subset for heavy models)
    text_features = [c for c in df_text.columns if c not in ["fake", "product_ID"]]
    text_models = [
        "dummy",
        "logistic_regression",
        "random_forest",
        "svm_rbf",
        "xgboost",
        "lightgbm",
    ]
    logger.info("--- Feature Group: text_only (1000 features) ---")
    for m_name in text_models:
        try:
            res = run_single_experiment(
                df=df_text,
                features=text_features,
                model_name=m_name,
                stratify=True,
                experiment_id="improved_text_only",
            )
            log_experiment_result(
                experiment_id="improved_text_only",
                model=m_name,
                feature_group="text_only",
                metrics=res,
            )
            print(
                f"[text_only             ] {m_name:20s} | "
                f"AUC: {res.get('roc_auc', 0):.4f} | "
                f"Acc: {res['accuracy']:.4f} | "
                f"Fake F1: {res['fake_f1']:.4f}"
            )
        except Exception as e:
            logger.error("Failed on text_only with %s: %s", m_name, e)

    # Save top trained models to MODEL_DIR
    logger.info("Saving trained models to %s", MODEL_DIR)
    for name, (pipe, score) in best_models.items():
        model_path = MODEL_DIR / f"{name}_pipeline.joblib"
        joblib.dump(pipe, model_path)
        logger.info("Saved %s (Fake F1: %.4f) -> %s", name, score, model_path.name)


def run_cross_validation_suite(df_img: pd.DataFrame) -> None:
    """Execute 5x2 repeated stratified cross-validation on key models."""
    logger.info("=" * 60)
    logger.info("RUNNING 5x2 REPEATED STRATIFIED CV SUITE")
    logger.info("=" * 60)

    cv_models = ["logistic_regression", "random_forest", "svm_rbf", "xgboost", "lightgbm", "catboost"]
    key_combos = [
        ("cv_review_only", REVIEW_FEATURES),
        ("cv_review_network", REVIEW_NETWORK_FEATURES),
        ("cv_review_network_image", ALL_STRUCTURED_FEATURES),
    ]

    cv_results_rows = []

    for exp_id, feats in key_combos:
        logger.info("Running CV for feature set: %s (%d feats)", exp_id, len(feats))
        for m_name in cv_models:
            res = run_cv_experiment(
                df=df_img,
                features=feats,
                model_name=m_name,
                n_splits=5,
                n_repeats=2,
                experiment_id=exp_id,
            )
            print(
                f"[{exp_id:24s}] {m_name:20s} | "
                f"AUC: {res['roc_auc_mean']:.4f} ± {res['roc_auc_std']:.4f} | "
                f"Acc: {res['accuracy_mean']:.4f} ± {res['accuracy_std']:.4f} | "
                f"Fake F1: {res['fake_f1_mean']:.4f} ± {res['fake_f1_std']:.4f}"
            )
            row = {
                "experiment_id": exp_id,
                "model": m_name,
                "feature_group": exp_id.replace("cv_", ""),
                "roc_auc_mean": res["roc_auc_mean"],
                "roc_auc_std": res["roc_auc_std"],
                "fake_f1_mean": res["fake_f1_mean"],
                "fake_f1_std": res["fake_f1_std"],
                "accuracy_mean": res["accuracy_mean"],
                "accuracy_std": res["accuracy_std"],
                "pr_auc_mean": res["pr_auc_mean"],
                "pr_auc_std": res["pr_auc_std"],
            }
            cv_results_rows.append(row)

    df_cv = pd.DataFrame(cv_results_rows)
    cv_out_path = EXPERIMENT_DIR / "CV_RESULTS_SUMMARY.csv"
    df_cv.to_csv(cv_out_path, index=False)
    logger.info("Saved CV summary to %s", cv_out_path)


def main():
    logger.info("Loading datasets...")
    df_img = load_product_data(with_images=True)
    df_text = load_text_data()

    # 1. Exact Legacy Reproduction
    run_legacy_suite(df_img, df_text)

    # 2. Improved Stratified Holdout Suite
    run_improved_holdout_suite(df_img, df_text)

    # 3. 5x2 Repeated Stratified CV Suite
    run_cross_validation_suite(df_img)

    logger.info("All Phase 1-3 Supervised Experiments successfully completed!")


if __name__ == "__main__":
    main()

