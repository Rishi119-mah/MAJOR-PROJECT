#!/usr/bin/env python3
"""
Verify every model in MODEL_REGISTRY:
- Model identifier
- Artifact path
- Load result
- Prediction result
- Probability support
- Calibration status
- Error, if any
"""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/mpl"
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

from src.config import MODEL_DIR
from src.api.main import MODEL_REGISTRY

def main():
    print("=" * 80)
    print("VERIFYING ALL REGISTERED MODEL ARTIFACTS IN MODEL_REGISTRY")
    print("=" * 80)

    rows = []
    all_ok = True

    for name, info in MODEL_REGISTRY.items():
        fname = info["filename"]
        path = MODEL_DIR / fname
        load_ok = False
        pred_ok = False
        prob_ok = False
        is_calibrated = "calibrated" in name
        err_msg = "None"

        if not path.exists():
            err_msg = f"File not found: {fname}"
            all_ok = False
            rows.append({
                "model_id": name,
                "artifact_path": str(path.relative_to(Path.cwd())),
                "load_status": "MISSING",
                "pred_status": "N/A",
                "prob_support": "N/A",
                "calibration": "Yes" if is_calibrated else "No",
                "error": err_msg,
            })
            continue

        try:
            model = joblib.load(path)
            load_ok = True
        except Exception as e:
            err_msg = f"Load error: {e}"
            all_ok = False
            rows.append({
                "model_id": name,
                "artifact_path": str(path.relative_to(Path.cwd())),
                "load_status": "FAILED",
                "pred_status": "N/A",
                "prob_support": "N/A",
                "calibration": "Yes" if is_calibrated else "No",
                "error": err_msg,
            })
            continue

        # Test prediction with schema
        try:
            req_feats = info["required_features"]
            df_sample = pd.DataFrame([{f: 0.5 for f in req_feats}])
            if "avg_review_rating" in df_sample.columns:
                df_sample["avg_review_rating"] = 4.2
            if "n_of_reviews" in df_sample.columns:
                df_sample["n_of_reviews"] = 50.0

            pred = model.predict(df_sample)
            pred_ok = True
            
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(df_sample)
                prob_ok = (probs.shape[1] == 2 and 0.0 <= probs[0, 1] <= 1.0)
            else:
                prob_ok = False

            rows.append({
                "model_id": name,
                "artifact_path": str(path.relative_to(Path.cwd())),
                "load_status": "OK",
                "pred_status": f"OK (class={int(pred[0])})",
                "prob_support": f"OK (p={probs[0, 1]:.4f})" if prob_ok else "No proba",
                "calibration": "Isotonic (Calibrated)" if is_calibrated else "Uncalibrated",
                "error": "None",
            })
        except Exception as e:
            err_msg = f"Inference error: {e}"
            all_ok = False
            rows.append({
                "model_id": name,
                "artifact_path": str(path.relative_to(Path.cwd())),
                "load_status": "OK",
                "pred_status": "FAILED",
                "prob_support": "FAILED",
                "calibration": "Yes" if is_calibrated else "No",
                "error": err_msg,
            })

    df_report = pd.DataFrame(rows)
    print(df_report.to_string(index=False))
    print("=" * 80)
    print(f"Total Models Verified: {len(rows)} | All Passed: {all_ok}")
    return all_ok

if __name__ == "__main__":
    main()

