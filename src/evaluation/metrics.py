"""
Evaluation metrics for the Fake Review Detection project.

Provides a single ``compute_all_metrics`` entry point that returns every
metric the project needs, plus formatting and persistence helpers.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute a comprehensive set of classification metrics.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth binary labels (0 or 1).
    y_pred : array-like of shape (n_samples,)
        Predicted binary labels.
    y_prob : array-like of shape (n_samples,) or None
        Predicted probability of the positive class (``fake=1``).
        If None, probability-based metrics (ROC-AUC, PR-AUC, Brier)
        are set to ``None``.

    Returns
    -------
    dict
        Keys:
          accuracy, fake_precision, fake_recall, fake_f1,
          genuine_precision, genuine_recall, genuine_f1,
          macro_f1, weighted_f1,
          specificity (class-0 recall),
          roc_auc, pr_auc, brier_score,
          confusion_matrix (2×2 list of lists).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Specificity = TN / (TN + FP), FPR = FP / (FP + TN)
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    results: dict[str, Any] = {
        # Overall
        "accuracy": float(accuracy_score(y_true, y_pred)),
        # Fake (positive class = 1)
        "fake_precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "fake_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "fake_f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        # Genuine (negative class = 0)
        "genuine_precision": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "genuine_recall": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "genuine_f1": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
        # Aggregated F1
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        # Specificity & False Positive Rate
        "specificity": specificity,
        "false_positive_rate": fpr,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        # Confusion matrix
        "confusion_matrix": cm.tolist(),
    }

    # Probability-based metrics
    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        results["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        results["pr_auc"] = float(average_precision_score(y_true, y_prob))
        results["brier_score"] = float(brier_score_loss(y_true, y_prob))

        # Expected Calibration Error (ECE) across 10 uniform bins
        bins = np.linspace(0.0, 1.0, 11)
        bin_indices = np.digitize(y_prob, bins) - 1
        bin_indices = np.clip(bin_indices, 0, 9)
        ece = 0.0
        n_total = len(y_true)
        for b in range(10):
            mask = bin_indices == b
            if np.any(mask):
                prop_b = np.sum(mask) / n_total
                acc_b = np.mean(y_true[mask])
                conf_b = np.mean(y_prob[mask])
                ece += prop_b * np.abs(acc_b - conf_b)
        results["expected_calibration_error"] = float(ece)
    else:
        results["roc_auc"] = None
        results["pr_auc"] = None
        results["brier_score"] = None
        results["expected_calibration_error"] = None

    return results


def format_metrics_table(results: dict[str, Any], title: str = "") -> str:
    """Return a human-readable formatted string of metrics.

    Parameters
    ----------
    results : dict
        Output of ``compute_all_metrics``.
    title : str, optional
        Header line for the table.

    Returns
    -------
    str
    """
    lines: list[str] = []
    if title:
        lines.append(f"{'=' * 60}")
        lines.append(f"  {title}")
        lines.append(f"{'=' * 60}")

    def _fmt(val: Any) -> str:
        if val is None:
            return "N/A"
        if isinstance(val, float):
            return f"{val:.4f}"
        return str(val)

    metric_order = [
        ("Accuracy", "accuracy"),
        ("Fake Precision", "fake_precision"),
        ("Fake Recall (Sensitivity)", "fake_recall"),
        ("Fake F1", "fake_f1"),
        ("Genuine Precision", "genuine_precision"),
        ("Genuine Recall (Specificity)", "specificity"),
        ("Genuine F1", "genuine_f1"),
        ("Macro F1", "macro_f1"),
        ("Weighted F1", "weighted_f1"),
        ("ROC-AUC", "roc_auc"),
        ("PR-AUC", "pr_auc"),
        ("Brier Score", "brier_score"),
    ]

    for label, key in metric_order:
        lines.append(f"  {label:<30s} {_fmt(results.get(key))}")

    cm = results.get("confusion_matrix")
    if cm is not None:
        lines.append("")
        lines.append("  Confusion Matrix (rows=true, cols=pred):")
        lines.append(f"    Pred Genuine  Pred Fake")
        lines.append(f"  Genuine  {cm[0][0]:>8d}  {cm[0][1]:>8d}")
        lines.append(f"  Fake     {cm[1][0]:>8d}  {cm[1][1]:>8d}")

    lines.append("")
    return "\n".join(lines)


def save_metrics(results: dict[str, Any], path: Path | str) -> None:
    """Persist metrics dict as JSON.

    Parameters
    ----------
    results : dict
        Output of ``compute_all_metrics`` (or any JSON-serialisable dict).
    path : Path or str
        Destination file (parent dirs created automatically).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types for JSON serialisation
    def _convert(obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    serialisable = {k: _convert(v) for k, v in results.items()}
    with open(path, "w") as fh:
        json.dump(serialisable, fh, indent=2, default=_convert)
    logger.info("Metrics saved to %s", path)
