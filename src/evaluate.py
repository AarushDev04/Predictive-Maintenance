"""
src/evaluate.py
Evaluation utilities used by the Prefect retrain flow
and the held-out validation cell (Section 18).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, classification_report,
)


def evaluate_model(
    model,
    X:          pd.DataFrame,
    y:          pd.Series,
    threshold:  float = 0.5,
    label:      str   = "Evaluation",
) -> dict:
    """
    Score a model on (X, y) and return a metrics dict.
    Prints a formatted summary.
    """
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)

    metrics = {
        "auc":       roc_auc_score(y, probs),
        "f1":        f1_score(y, preds,  zero_division=0),
        "recall":    recall_score(y, preds, zero_division=0),
        "precision": precision_score(y, preds, zero_division=0),
        "n_rows":    len(y),
        "n_positive":int(y.sum()),
        "threshold": threshold,
    }

    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    print(f"  Rows:      {metrics['n_rows']:,}")
    print(f"  Positives: {metrics['n_positive']:,}  "
          f"({metrics['n_positive']/max(metrics['n_rows'],1)*100:.2f}%)")
    print(f"  AUC-ROC:   {metrics['auc']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"{'='*55}")
    print(classification_report(
        y, preds,
        target_names=["No Failure", "Failure in 48h"],
        zero_division=0,
    ))

    return metrics


def find_optimal_threshold(
    y_true:        pd.Series,
    probs:         np.ndarray,
    recall_target: float = 0.90,
) -> float:
    """
    Return the highest threshold that still achieves recall >= recall_target.
    Falls back to 0.5 if no such threshold exists.
    """
    from sklearn.metrics import precision_recall_curve
    precision_arr, recall_arr, thresholds = precision_recall_curve(y_true, probs)

    valid = np.where(recall_arr[:-1] >= recall_target)[0]
    if len(valid) == 0:
        print(f"⚠️  No threshold achieves recall ≥ {recall_target} — using 0.5")
        return 0.5

    opt_idx   = valid[np.argmax(precision_arr[valid])]
    opt_thresh = float(thresholds[opt_idx])
    print(f"Optimal threshold (recall≥{recall_target}): {opt_thresh:.3f}")
    print(f"  Precision @ threshold: {precision_arr[opt_idx]:.3f}")
    print(f"  Recall    @ threshold: {recall_arr[opt_idx]:.3f}")
    return opt_thresh