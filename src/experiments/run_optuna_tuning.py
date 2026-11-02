"""
Run Optuna hyperparameter tuning for Random Forest and XGBoost on review_network_image.
Saves best hyperparameters to outputs/experiments/BEST_HYPERPARAMETERS.json
and trains the best tuned models.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
import joblib

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import ALL_STRUCTURED_FEATURES, EXPERIMENT_DIR, MODEL_DIR, TARGET
from src.data.data_loader import load_product_data
from src.models.tuning import tune_random_forest, tune_xgboost, tune_lightgbm
from src.models.pipeline import run_single_experiment, log_experiment_result

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("optuna_tuning")


def main():
    logger.info("Loading product dataset with image features...")
    df = load_product_data(with_images=True)
    X = df[ALL_STRUCTURED_FEATURES]
    y = df[TARGET]

    logger.info("Tuning Random Forest (20 trials)...")
    res_rf = tune_random_forest(X, y, n_trials=20, scoring="fake_f1")

    logger.info("Tuning XGBoost (20 trials)...")
    res_xgb = tune_xgboost(X, y, n_trials=20, scoring="fake_f1")

    logger.info("Tuning LightGBM (20 trials)...")
    res_lgb = tune_lightgbm(X, y, n_trials=20, scoring="fake_f1")

    best_hyperparams = {
        "random_forest": res_rf,
        "xgboost": res_xgb,
        "lightgbm": res_lgb,
    }

    out_json = EXPERIMENT_DIR / "BEST_HYPERPARAMETERS.json"
    with open(out_json, "w") as f:
        json.dump(best_hyperparams, f, indent=2)
    logger.info("Saved best hyperparameters to %s", out_json)

    # Evaluate tuned models on holdout
    logger.info("Evaluating tuned models on holdout...")
    for name, res in [("random_forest", res_rf), ("xgboost", res_xgb), ("lightgbm", res_lgb)]:
        params = res["params"]
        eval_res = run_single_experiment(
            df=df,
            features=ALL_STRUCTURED_FEATURES,
            model_name=name,
            stratify=True,
            experiment_id=f"tuned_{name}",
            **params,
        )
        log_experiment_result(
            experiment_id=f"tuned_{name}",
            model=f"tuned_{name}",
            feature_group="review_network_image",
            metrics=eval_res,
        )
        print(
            f"[TUNED {name:15s}] AUC: {eval_res.get('roc_auc', 0):.4f} | "
            f"PR-AUC: {eval_res.get('pr_auc', 0):.4f} | "
            f"Acc: {eval_res['accuracy']:.4f} | "
            f"Fake F1: {eval_res['fake_f1']:.4f}"
        )
        # Save tuned model
        joblib.dump(eval_res["pipeline"], MODEL_DIR / f"tuned_{name}_pipeline.joblib")


if __name__ == "__main__":
    main()

