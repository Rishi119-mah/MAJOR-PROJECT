"""
Hyperparameter tuning using Optuna for Random Forest, XGBoost, and LightGBM.

All tuning evaluates candidates strictly inside stratified folds to prevent leakage.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

from src.config import MODEL_DIR, RANDOM_SEED, TARGET

# Suppress Optuna log verbosity
optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)


def tune_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 20,
    scoring: str = "fake_f1",
) -> dict[str, Any]:
    """Tune RandomForestClassifier using Optuna over 5-fold CV."""
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
            "max_depth": trial.suggest_int("max_depth", 10, 45, step=5),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 8),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4),
            "max_features": "sqrt",
            "random_state": RANDOM_SEED,
            "n_jobs": -1,
        }
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        scores = []
        for tr_idx, val_idx in skf.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

            clf = RandomForestClassifier(**params)
            clf.fit(X_tr, y_tr)

            if scoring == "fake_f1":
                y_pred = clf.predict(X_val)
                s = f1_score(y_val, y_pred, pos_label=1, zero_division=0)
            else:
                y_prob = clf.predict_proba(X_val)[:, 1]
                s = roc_auc_score(y_val, y_prob)
            scores.append(s)
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_params["random_state"] = RANDOM_SEED
    best_params["n_jobs"] = -1
    logger.info("Best RF Params (%s=%.4f): %s", scoring, study.best_value, best_params)
    return {"params": best_params, "best_score": study.best_value}


def tune_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 20,
    scoring: str = "fake_f1",
) -> dict[str, Any]:
    """Tune XGBClassifier using Optuna over 5-fold CV."""
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 250, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0, step=0.1),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0, step=0.1),
            "eval_metric": "logloss",
            "random_state": RANDOM_SEED,
        }
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        scores = []
        for tr_idx, val_idx in skf.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

            clf = xgb.XGBClassifier(**params)
            clf.fit(X_tr, y_tr)

            if scoring == "fake_f1":
                y_pred = clf.predict(X_val)
                s = f1_score(y_val, y_pred, pos_label=1, zero_division=0)
            else:
                y_prob = clf.predict_proba(X_val)[:, 1]
                s = roc_auc_score(y_val, y_prob)
            scores.append(s)
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_params["random_state"] = RANDOM_SEED
    best_params["eval_metric"] = "logloss"
    logger.info("Best XGB Params (%s=%.4f): %s", scoring, study.best_value, best_params)
    return {"params": best_params, "best_score": study.best_value}


def tune_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 20,
    scoring: str = "fake_f1",
) -> dict[str, Any]:
    """Tune LGBMClassifier using Optuna over 5-fold CV."""
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 250, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0, step=0.1),
            "random_state": RANDOM_SEED,
            "verbose": -1,
        }
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        scores = []
        for tr_idx, val_idx in skf.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

            clf = lgb.LGBMClassifier(**params)
            clf.fit(X_tr, y_tr)

            if scoring == "fake_f1":
                y_pred = clf.predict(X_val)
                s = f1_score(y_val, y_pred, pos_label=1, zero_division=0)
            else:
                y_prob = clf.predict_proba(X_val)[:, 1]
                s = roc_auc_score(y_val, y_prob)
            scores.append(s)
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_params["random_state"] = RANDOM_SEED
    best_params["verbose"] = -1
    logger.info("Best LightGBM Params (%s=%.4f): %s", scoring, study.best_value, best_params)
    return {"params": best_params, "best_score": study.best_value}

