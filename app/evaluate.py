"""
routers/evaluate.py

POST /evaluate          — Run evaluation against a results.jsonl and ground truth
GET  /evaluate/latest   — Return the most recently cached evaluation result
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.evaluator import run_evaluation
from app.schemas.models import EvaluationResponse
from app.config import get_settings

router = APIRouter()
settings = get_settings()

_latest_result: dict = {}


class EvaluateRequest(BaseModel):
    results_jsonl_path: str = ""   # path on the server; empty = use default
    dataset_root: str = ""         # empty = use settings default


# ─────────────────────────────────────────────────────────────────────────────
# POST /evaluate
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "",
    response_model=EvaluationResponse,
    summary="Evaluate pipeline predictions against ground truth",
    description=(
        "Compares a results.jsonl file (produced by batch analysis) against the "
        "JSON sidecar ground truth. Returns accuracy, macro-F1, per-class P/R/F1, "
        "confusion matrix, and critical recall."
    ),
)
def evaluate(req: EvaluateRequest = None):
    if req is None:
        req = EvaluateRequest()

    results_path = req.results_jsonl_path or settings.default_results_jsonl
    dataset_root = req.dataset_root or settings.dataset_root

    if not results_path:
        raise HTTPException(
            status_code=400,
            detail="No results_jsonl_path provided and no default configured.",
        )

    try:
        result = run_evaluation(results_path=results_path, dataset_root=dataset_root)
        _latest_result.update(result)
        return EvaluationResponse(**result)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# GET /evaluate/latest
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/latest",
    response_model=EvaluationResponse,
    summary="Return the most recent evaluation result",
)
def latest_evaluation():
    if not _latest_result:
        raise HTTPException(
            status_code=404,
            detail="No evaluation has been run yet. POST /evaluate first.",
        )
    return EvaluationResponse(**_latest_result)
