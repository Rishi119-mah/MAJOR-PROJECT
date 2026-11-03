"""
Run Optuna hyperparameter tuning for Random Forest, XGBoost, and LightGBM.

LEAKAGE-FREE EVALUATION PROTOCOL:
1. Strict Stratified Split into Development Set (80%) and Final Test Set (20%).
2. Optuna tunes hyperparameters strictly on the Development Set using 5-fold CV.
   The final test set is never seen by Optuna.
3. Best hyperparameters are trained on the full Development Set.
4. Final unbiased evaluation is performed on the untouched Test Set.
5. Hyperparameters and evaluation metrics are persisted.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import ALL_STRUCTURED_FEATURES, EXPERIMENT_DIR, MODEL_DIR, TARGET, RANDOM_SEED
from src.data.data_loader import load_product_data
from src.models.tuning import tune_random_forest, tune_xgboost, tune_lightgbm
from src.evaluation.metrics import compute_all_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("optuna_tuning")


def main():
    logger.info("Loading product dataset with image features...")
    df = load_product_data(with_images=True)
    features = ALL_STRUCTURED_FEATURES

    # 1. Permanent Stratified Split: Development (80%) vs Final Test (20%)
    logger.info("Splitting dataset into Development (80%) and Final Test (20%)...")
    df_dev, df_test = train_test_split(
        df, test_size=0.20, random_state=RANDOM_SEED, stratify=df[TARGET]
    )

    X_dev = df_dev[features].copy()
    y_dev = df_dev[TARGET].copy()

    X_test = df_test[features].copy()
    y_test = df_test[TARGET].copy()

    logger.info("Development samples: %d (pos=%d), Test samples: %d (pos=%d)",
                len(df_dev), y_dev.sum(), len(df_test), y_test.sum())

    # 2. Tune ONLY on Development Data
    logger.info("Tuning Random Forest on Development Data (20 trials)...")
    res_rf = tune_random_forest(X_dev, y_dev, n_trials=20, scoring="fake_f1")

    logger.info("Tuning XGBoost on Development Data (20 trials)...")
    res_xgb = tune_xgboost(X_dev, y_dev, n_trials=20, scoring="fake_f1")

    logger.info("Tuning LightGBM on Development Data (20 trials)...")
    res_lgb = tune_lightgbm(X_dev, y_dev, n_trials=20, scoring="fake_f1")

    best_hyperparams = {
        "metadata": {
            "protocol": "Strict 80/20 train/test split. Tuning strictly on 80% dev data via 5-fold CV.",
            "dev_samples": len(df_dev),
            "test_samples": len(df_test),
            "random_seed": RANDOM_SEED,
            "feature_set": "review_network_image (28 features)",
        },
        "random_forest": res_rf,
        "xgboost": res_xgb,
        "lightgbm": res_lgb,
    }

    out_json = EXPERIMENT_DIR / "BEST_HYPERPARAMETERS.json"
    with open(out_json, "w") as f:
        json.dump(best_hyperparams, f, indent=2)
    logger.info("Saved best hyperparameters to %s", out_json)

    # 3. Train on Development Data, Evaluate on Final Test Set
    logger.info("Training tuned models on full Development data and evaluating on Final Test...")
    model_factories = {
        "random_forest": lambda p: RandomForestClassifier(**p),
        "xgboost": lambda p: xgb.XGBClassifier(**p),
        "lightgbm": lambda p: lgb.LGBMClassifier(**p),
    }

    final_eval_results = []
    for name, res in [("random_forest", res_rf), ("xgboost", res_xgb), ("lightgbm", res_lgb)]:
        params = res["params"]
        model = model_factories[name](params)

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", model),
        ])

        # Fit strictly on dev
        pipeline.fit(X_dev, y_dev)

        # Evaluate strictly on test
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None

        metrics = compute_all_metrics(y_test, y_pred, y_prob)
        metrics["model"] = f"tuned_{name}"
        metrics["feature_group"] = "review_network_image"
        metrics["cv_dev_score"] = res["best_score"]
        final_eval_results.append(metrics)

        print(
            f"[LEAKAGE-FREE TUNED {name:15s}] Dev CV F1: {res['best_score']:.4f} | "
            f"Test AUC: {metrics.get('roc_auc', 0):.4f} | "
            f"Test PR-AUC: {metrics.get('pr_auc', 0):.4f} | "
            f"Test Acc: {metrics['accuracy']:.4f} | "
            f"Test Fake F1: {metrics['fake_f1']:.4f}"
        )

        # Save tuned pipeline
        save_path = MODEL_DIR / f"tuned_{name}_pipeline.joblib"
        joblib.dump(pipeline, save_path)
        logger.info("Saved tuned pipeline to %s", save_path)

    # Save comparison to CSV
    df_eval = pd.DataFrame(final_eval_results)
    eval_csv = EXPERIMENT_DIR / "TUNED_MODELS_FINAL_TEST_EVAL.csv"
    df_eval.to_csv(eval_csv, index=False)
    logger.info("Saved final test evaluation to %s", eval_csv)


if __name__ == "__main__":
    main()
