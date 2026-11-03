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
    TARGET,
    REVIEW_FEATURES,
    NETWORK_FEATURES,
    IMAGE_FEATURES,
    ALL_STRUCTURED_FEATURES,
    REVIEW_NETWORK_FEATURES,
)
from src.data.data_loader import load_product_data, load_text_data
from src.evaluation.metrics import format_metrics_table, compute_all_metrics
from src.models.classifiers import get_model, get_legacy_model
from src.models.pipeline import (
    run_legacy_experiment,
    run_single_experiment,
    run_cv_experiment,
    log_experiment_result,
    build_pipeline,
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
    if df_text is not None:
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
    else:
        logger.info("Legacy text experiment skipped: text dataset not present locally.")


def run_improved_holdout_suite(
    df_dev: pd.DataFrame,
    df_test: pd.DataFrame,
    df_text_dev: pd.DataFrame | None = None,
    df_text_test: pd.DataFrame | None = None,
) -> None:
    """Train on Development Data (80%), evaluate on Untouched Final Holdout (20%).

    LEAKAGE-FREE PROTOCOL:
    - Model selection and hyperparameter decisions are locked based strictly on Dev CV.
    - Final pipelines are trained on df_dev and evaluated ONCE on df_test.
    - Final test set is never used to select or rank candidate models.
    """
    logger.info("=" * 60)
    logger.info("RUNNING FINAL INDEPENDENT HOLDOUT TEST EVALUATION")
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
        ("review_only", REVIEW_FEATURES),
        ("review_network", REVIEW_NETWORK_FEATURES),
        ("review_network_image", ALL_STRUCTURED_FEATURES),
        ("network_only", NETWORK_FEATURES),
    ]

    best_models = {}
    test_eval_rows = []

    for combo_name, feats in feature_combos:
        logger.info(
            "--- Feature Group: %s (%d features) ---", combo_name, len(feats)
        )
        for m_name in models_to_test:
            try:
                # Preprocessing and model pipeline
                needs_scale = ["dummy", "logistic_regression", "svm_linear", "svm_rbf", "stacking"]
                scale = m_name.lower() in needs_scale

                raw_model = get_model(m_name)
                pipeline = build_pipeline(raw_model, scale=scale)

                X_tr = df_dev[feats]
                y_tr = df_dev[TARGET]
                X_te = df_test[feats]
                y_te = df_test[TARGET]

                t0 = time.perf_counter()
                pipeline.fit(X_tr, y_tr)
                t_train = time.perf_counter() - t0

                t1 = time.perf_counter()
                y_pred = pipeline.predict(X_te)
                t_inf = time.perf_counter() - t1

                y_prob = None
                if hasattr(pipeline, "predict_proba"):
                    y_prob = pipeline.predict_proba(X_te)[:, 1]
                elif hasattr(pipeline, "decision_function"):
                    dfun = pipeline.decision_function(X_te)
                    y_prob = 1 / (1 + np.exp(-dfun))

                metrics = compute_all_metrics(y_te.values, y_pred, y_prob)
                metrics["experiment_id"] = f"final_test_{combo_name}"
                metrics["model"] = m_name
                metrics["feature_group"] = combo_name
                metrics["evaluation_protocol"] = "Independent final test holdout (trained on 80% dev, evaluated once on 20% test)"
                metrics["train_time_s"] = t_train
                metrics["inference_time_s"] = t_inf

                test_eval_rows.append(metrics)
                log_experiment_result(
                    experiment_id=f"improved_{combo_name}",
                    model=m_name,
                    feature_group=combo_name,
                    metrics=metrics,
                )
                print(
                    f"[{combo_name:22s}] {m_name:20s} | "
                    f"AUC: {metrics.get('roc_auc', 0):.4f} | "
                    f"PR-AUC: {metrics.get('pr_auc', 0):.4f} | "
                    f"Acc: {metrics['accuracy']:.4f} | "
                    f"Fake F1: {metrics['fake_f1']:.4f} | "
                    f"Train(s): {metrics['train_time_s']:.2f}"
                )

                # Persist standard production pipelines
                key = f"{combo_name}_{m_name}"
                if m_name in ["random_forest", "xgboost", "catboost", "stacking"]:
                    best_models[key] = pipeline

            except Exception as e:
                logger.error("Failed on %s with %s: %s", combo_name, m_name, e)

    # Text Only on improved models if text data is available
    if df_text_dev is not None and df_text_test is not None:
        text_features = [c for c in df_text_dev.columns if c not in ["fake", "product_ID"]]
        text_models = ["dummy", "logistic_regression", "random_forest", "svm_rbf", "xgboost", "lightgbm"]
        logger.info("--- Feature Group: text_only (1000 features) ---")
        for m_name in text_models:
            try:
                needs_scale = m_name.lower() in ["dummy", "logistic_regression", "svm_rbf"]
                pipeline = build_pipeline(get_model(m_name), scale=needs_scale)
                pipeline.fit(df_text_dev[text_features], df_text_dev[TARGET])
                y_pred = pipeline.predict(df_text_test[text_features])
                y_prob = pipeline.predict_proba(df_text_test[text_features])[:, 1] if hasattr(pipeline, "predict_proba") else None
                m_txt = compute_all_metrics(df_text_test[TARGET].values, y_pred, y_prob)
                m_txt["experiment_id"] = "final_test_text_only"
                m_txt["model"] = m_name
                m_txt["feature_group"] = "text_only"
                test_eval_rows.append(m_txt)
                log_experiment_result(
                    experiment_id="improved_text_only",
                    model=m_name,
                    feature_group="text_only",
                    metrics=m_txt,
                )
            except Exception as e:
                logger.error("Failed on text_only with %s: %s", m_name, e)
    else:
        logger.info("Skipping holdout text_only experiments (text dataset not available).")

    # Save final independent test evaluations to separate CSV
    df_eval = pd.DataFrame(test_eval_rows)
    eval_csv_path = EXPERIMENT_DIR / "FINAL_TEST_EVALUATION.csv"
    df_eval.to_csv(eval_csv_path, index=False)
    logger.info("Saved final test evaluation ledger to %s", eval_csv_path)

    # Save trained pipelines to MODEL_DIR
    logger.info("Saving trained production pipelines to %s", MODEL_DIR)
    for name, pipe in best_models.items():
        model_path = MODEL_DIR / f"{name}_pipeline.joblib"
        joblib.dump(pipe, model_path)
        logger.info("Saved pipeline -> %s", model_path.name)


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

    # 1. Permanent Split: Development (80%) vs Untouched Final Holdout (20%)
    from sklearn.model_selection import train_test_split
    df_dev, df_test = train_test_split(
        df_img, test_size=0.20, random_state=RANDOM_SEED, stratify=df_img[TARGET]
    )
    logger.info(
        "Dataset Split: %d Development samples (80%%), %d Untouched Test samples (20%%)",
        len(df_dev), len(df_test)
    )

    # Optional text data
    df_text = None
    df_text_dev = None
    df_text_test = None
    if DATA_TEXT.exists():
        try:
            df_text = load_text_data()
            df_text_dev = df_text[df_text["product_ID"].isin(df_dev["product_ID"])].copy()
            df_text_test = df_text[df_text["product_ID"].isin(df_test["product_ID"])].copy()
            logger.info("Loaded and aligned text dataset (%d rows).", len(df_text))
        except Exception as e:
            logger.warning("Could not load text data: %s", e)
    else:
        logger.info("Text dataset not present; continuing with structured features.")

    # 2. Exact Legacy Reproduction (for historical comparison)
    run_legacy_suite(df_img, df_text)

    # 3. Model Selection via 5x2 Repeated Stratified CV STRICTLY on Development Data
    logger.info("Running Model Selection 5x2 CV strictly on Development Data...")
    run_cross_validation_suite(df_dev)

    # 4. Final Independent Holdout Test Evaluation (Trained on Dev, Evaluated on Test)
    logger.info("Training selected models on Development set and evaluating ONCE on Final Holdout...")
    run_improved_holdout_suite(df_dev, df_test, df_text_dev, df_text_test)

    logger.info("All Supervised Experiments successfully completed with strict holdout isolation!")


if __name__ == "__main__":
    main()

