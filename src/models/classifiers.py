"""
Classifier definitions and model factory for Fake Review Detection.

Provides standard interfaces for instantiating baseline, legacy, and improved models.
"""

from __future__ import annotations

import logging
from typing import Any

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb

from src.config import RANDOM_SEED

logger = logging.getLogger(__name__)

# Check optional libraries
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False


def get_model(name: str, **kwargs: Any):
    """Factory function to instantiate models for improved experiments.

    Parameters
    ----------
    name : str
        Model identifier.
    kwargs : Any
        Hyperparameters to override defaults.

    Returns
    -------
    BaseEstimator
        Configured scikit-learn compatible classifier.
    """
    name_clean = name.lower().strip()

    if name_clean == "dummy":
        params = {"strategy": "most_frequent"}
        params.update(kwargs)
        return DummyClassifier(**params)

    elif name_clean == "logistic_regression":
        params = {
            "max_iter": 1000,
            "random_state": RANDOM_SEED,
            "solver": "lbfgs",
            "class_weight": None,
        }
        params.update(kwargs)
        return LogisticRegression(**params)

    elif name_clean == "random_forest":
        params = {
            "n_estimators": 100,
            "max_depth": 40,
            "min_samples_split": 6,
            "min_samples_leaf": 3,
            "max_features": "sqrt",
            "bootstrap": True,
            "random_state": RANDOM_SEED,
            "n_jobs": -1,
        }
        params.update(kwargs)
        return RandomForestClassifier(**params)

    elif name_clean == "svm_linear":
        params = {
            "kernel": "linear",
            "probability": True,
            "random_state": RANDOM_SEED,
        }
        params.update(kwargs)
        return SVC(**params)

    elif name_clean == "svm_rbf":
        params = {
            "kernel": "rbf",
            "probability": True,
            "random_state": RANDOM_SEED,
        }
        params.update(kwargs)
        return SVC(**params)

    elif name_clean == "xgboost":
        params = {
            "eval_metric": "logloss",
            "random_state": RANDOM_SEED,
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "use_label_encoder": False,
        }
        params.update(kwargs)
        return xgb.XGBClassifier(**params)

    elif name_clean == "lightgbm":
        if not HAS_LIGHTGBM:
            raise ImportError("lightgbm is not installed.")
        params = {
            "random_state": RANDOM_SEED,
            "n_estimators": 100,
            "learning_rate": 0.1,
            "verbose": -1,
        }
        params.update(kwargs)
        return lgb.LGBMClassifier(**params)

    elif name_clean == "catboost":
        if not HAS_CATBOOST:
            raise ImportError("catboost is not installed.")
        params = {
            "random_state": RANDOM_SEED,
            "iterations": 200,
            "learning_rate": 0.1,
            "verbose": 0,
        }
        params.update(kwargs)
        return cb.CatBoostClassifier(**params)

    elif name_clean == "stacking":
        base_estimators = [
            ("lr", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=20,
                    min_samples_split=6,
                    min_samples_leaf=3,
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
                ),
            ),
            ("svc", SVC(kernel="rbf", probability=True, random_state=RANDOM_SEED)),
            (
                "xgb",
                xgb.XGBClassifier(
                    eval_metric="logloss",
                    random_state=RANDOM_SEED,
                    n_estimators=100,
                    max_depth=6,
                ),
            ),
        ]
        meta_classifier = LogisticRegression(random_state=RANDOM_SEED)
        return StackingClassifier(
            estimators=base_estimators,
            final_estimator=meta_classifier,
            cv=5,
            n_jobs=-1,
        )

    else:
        raise ValueError(f"Unknown model name: {name}")


def get_legacy_model(name: str):
    """Instantiate a model with the EXACT parameters from the original classification.py.

    Parameters
    ----------
    name : str
        Model identifier.

    Returns
    -------
    BaseEstimator
        Legacy scikit-learn classifier.
    """
    name_clean = name.lower().strip()

    if name_clean == "logistic_regression":
        return LogisticRegression(max_iter=400)

    elif name_clean == "random_forest":
        return RandomForestClassifier(
            random_state=42,
            n_estimators=100,
            min_samples_leaf=3,
            min_samples_split=6,
            max_features="sqrt",
            max_depth=40,
            bootstrap=True,
            n_jobs=-1,
        )

    elif name_clean == "svm_linear":
        return SVC(kernel="linear", probability=True)

    elif name_clean == "xgboost":
        return xgb.XGBClassifier()

    else:
        raise ValueError(f"Unknown legacy model name: {name}")

