"""
core/evaluator.py — Thin wrapper around evaluate_pipeline.py logic.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.evaluate_pipeline import load_ground_truth, read_results_jsonl  # noqa: E402

RISK_TIERS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


def run_evaluation(results_path: str, dataset_root: str) -> Dict[str, Any]:
    """
    Evaluate a results.jsonl file against ground truth JSON sidecars.
    Returns structured metrics dict matching EvaluationResponse schema.
    """
    rp = Path(results_path).resolve()
    dr = Path(dataset_root).resolve()

    gt_map = load_ground_truth(dr)
    rows = read_results_jsonl(rp)

    y_true: List[str] = []
    y_pred: List[str] = []
    alerts_present: List[int] = []
    missing_gt = 0

    for r in rows:
        pdf = r.get("pdf")
        pred = r.get("risk_level")
        if pred not in RISK_TIERS:
            continue
        gt = gt_map.get(pdf)
        if gt not in RISK_TIERS:
            missing_gt += 1
            continue
        y_true.append(gt)
        y_pred.append(pred)
        alerts_present.append(1 if (r.get("alerts") or []) else 0)

    if not y_true:
        raise ValueError(
            "No matched predictions found. Check that results.jsonl filenames "
            "match the JSON sidecar naming convention."
        )

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=RISK_TIERS)
    pr, rc, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=RISK_TIERS, zero_division=0
    )
    critical_recall = rc[RISK_TIERS.index("CRITICAL")]
    alert_rate = float(np.mean(alerts_present)) if alerts_present else 0.0

    # alert rate by predicted tier
    alert_by_tier = {t: {"count": 0, "alerts": 0} for t in RISK_TIERS}
    for _, pred, a in zip(y_true, y_pred, alerts_present):
        alert_by_tier[pred]["count"] += 1
        alert_by_tier[pred]["alerts"] += int(a)

    per_tier = {}
    for i, t in enumerate(RISK_TIERS):
        per_tier[t] = {
            "precision": float(pr[i]),
            "recall": float(rc[i]),
            "f1": float(f1[i]),
            "support": int(sup[i]),
        }

    # confusion matrix as nested dict for JSON serialisation
    cm_dict = {}
    for i, true_t in enumerate(RISK_TIERS):
        cm_dict[true_t] = {RISK_TIERS[j]: int(cm[i, j]) for j in range(len(RISK_TIERS))}

    return {
        "matched_samples": len(y_true),
        "skipped_missing_gt": missing_gt,
        "accuracy": float(acc),
        "critical_recall": float(critical_recall),
        "alert_rate": alert_rate,
        "per_tier": per_tier,
        "confusion_matrix": cm_dict,
        "alert_rate_by_tier": alert_by_tier,
    }
