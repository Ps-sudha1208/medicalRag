#!/usr/bin/env python3
"""
Evaluate pipeline results against synthetic ground truth.

Inputs:
- results.jsonl produced by run_folder (one JSON per PDF)
- dataset_root containing jsons/ with ground truth sidecars

Outputs:
- overall metrics: accuracy, macro-F1, weighted-F1
- per-class precision/recall/F1/support
- confusion matrix
- alert stats (overall + by predicted tier)
- critical recall emphasis (recall for CRITICAL class)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

RISK_TIERS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

def parse_filename_for_pid_pod(name: str) -> Tuple[Optional[str], Optional[int]]:
    # Example: PT0002_POD17_LOW_20260206.pdf
    pid = None
    pod = None
    import re
    m1 = re.search(r"(PT\d{4})", name)
    m2 = re.search(r"_POD(\d+)", name)
    if m1:
        pid = m1.group(1)
    if m2:
        pod = int(m2.group(1))
    return pid, pod

def load_ground_truth(dataset_root: Path) -> Dict[str, str]:
    """
    Returns mapping: pdf_filename -> gt_risk_level
    Uses json sidecars under dataset_root/jsons
    """
    gt_map: Dict[str, str] = {}
    json_dir = dataset_root / "jsons"
    if not json_dir.exists():
        raise FileNotFoundError(f"Missing jsons dir: {json_dir}")

    for jp in json_dir.glob("*.json"):
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
            gt = data.get("ground_truth", {}) or {}
            # Prefer explicit ground truth key if present
            y = gt.get("risk_level_rule") or gt.get("risk_level") or data.get("risk_level")
            if y not in RISK_TIERS:
                # As fallback, try daily_observation key used earlier
                dob = data.get("daily_observation", {}) or {}
                y = dob.get("rule_risk_level") or dob.get("risk_level")
            if y not in RISK_TIERS:
                continue

            # Link json sidecar to pdf name by convention
            # If you used same stem as pdf, this should work:
            pdf_name_guess = jp.with_suffix(".pdf").name
            gt_map[pdf_name_guess] = y
        except Exception:
            continue

    return gt_map

def read_results_jsonl(results_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with results_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def tier_index(t: str) -> int:
    return RISK_TIERS.index(t) if t in RISK_TIERS else -1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=str, required=True, help="Path to results.jsonl")
    ap.add_argument("--dataset-root", type=str, required=True, help="Dataset root containing jsons/")
    ap.add_argument("--gt-key", type=str, default=None,
                    help="Optional explicit ground truth key inside ground_truth (e.g., risk_level_rule)")
    args = ap.parse_args()

    results_path = Path(args.results).resolve()
    dataset_root = Path(args.dataset_root).resolve()

    if not results_path.exists():
        raise FileNotFoundError(f"results.jsonl not found: {results_path}")

    gt_map = load_ground_truth(dataset_root)
    if not gt_map:
        raise RuntimeError("No ground truth found. Check dataset_root/jsons and gt fields.")

    rows = read_results_jsonl(results_path)
    if not rows:
        raise RuntimeError("No rows found in results.jsonl")

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
        raise RuntimeError("No matched predictions with ground truth. Check filename mapping conventions.")

    # Overall metrics
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=RISK_TIERS)

    # Per-class metrics
    pr, rc, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=RISK_TIERS, zero_division=0
    )

    # Critical recall emphasis
    critical_recall = rc[RISK_TIERS.index("CRITICAL")]

    # Alert stats
    alert_rate = float(np.mean(alerts_present)) if alerts_present else 0.0

    # Alert rate by predicted tier
    alert_by_tier = {t: {"count": 0, "alerts": 0} for t in RISK_TIERS}
    for gt, pred, a in zip(y_true, y_pred, alerts_present):
        alert_by_tier[pred]["count"] += 1
        alert_by_tier[pred]["alerts"] += int(a)

    # Print summary
    print("\n================ EVALUATION SUMMARY ================\n")
    print(f"Matched samples: {len(y_true)}")
    if missing_gt:
        print(f"Skipped (missing GT mapping): {missing_gt}")

    print(f"\nAccuracy: {acc:.4f}")
    print(f"Critical Recall (CRITICAL class): {critical_recall:.4f}")
    print(f"Alert rate (any alert): {alert_rate:.4f}")

    print("\nPer-tier metrics:")
    for i, t in enumerate(RISK_TIERS):
        print(f"  {t:8s}  P={pr[i]:.3f}  R={rc[i]:.3f}  F1={f1[i]:.3f}  support={sup[i]}")

    print("\nConfusion Matrix (rows=true, cols=pred):")
    header = "         " + "  ".join([f"{t:8s}" for t in RISK_TIERS])
    print(header)
    for i, t in enumerate(RISK_TIERS):
        row = "  ".join([f"{cm[i, j]:8d}" for j in range(len(RISK_TIERS))])
        print(f"{t:8s}  {row}")

    print("\nAlert rate by predicted tier:")
    for t in RISK_TIERS:
        c = alert_by_tier[t]["count"]
        a = alert_by_tier[t]["alerts"]
        rate = (a / c) if c else 0.0
        print(f"  {t:8s}: {rate:.3f} ({a}/{c})")

    print("\nFull classification_report:")
    print(classification_report(y_true, y_pred, labels=RISK_TIERS, zero_division=0))

    print("\n====================================================\n")

if __name__ == "__main__":
    main()
