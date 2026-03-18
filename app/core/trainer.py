"""
core/trainer.py — Wrapper around the Random Forest training logic.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.clinical_langgraph_pipeline import train_random_forest  # noqa: E402


def train_model(dataset_root: str, rf_path: str) -> Dict[str, Any]:
    """
    Train the Random Forest classifier on JSON ground truth sidecars
    and save the model to rf_path.
    Returns a status dict.
    """
    dr = Path(dataset_root).resolve()
    out = Path(rf_path).resolve()

    train_random_forest(dr, out)

    return {
        "status": "ok",
        "model_path": str(out),
        "message": f"Model trained and saved to {out}",
    }


def get_model_status(rf_path: str) -> Dict[str, Any]:
    p = Path(rf_path).resolve()
    exists = p.exists()
    return {
        "model_exists": exists,
        "model_path": str(p),
        "message": "Model file found." if exists else "No trained model found. POST /model/train to create one.",
    }