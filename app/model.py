"""
routers/model.py

POST /model/train   — Train the Random Forest classifier
GET  /model/status  — Check whether a trained model exists
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks

from app.core.trainer import train_model, get_model_status
from app.schemas.models import ModelTrainResponse, ModelStatusResponse
from app.config import get_settings

router = APIRouter()
settings = get_settings()

_train_status: dict = {}


# ─────────────────────────────────────────────────────────────────────────────
# GET /model/status
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/status",
    response_model=ModelStatusResponse,
    summary="Check if a trained ML model exists",
)
def model_status():
    return get_model_status(settings.rf_path)


# ─────────────────────────────────────────────────────────────────────────────
# POST /model/train
# ─────────────────────────────────────────────────────────────────────────────

def _do_train():
    _train_status["status"] = "running"
    try:
        result = train_model(
            dataset_root=settings.dataset_root,
            rf_path=settings.rf_path,
        )
        _train_status["status"] = "done"
        _train_status["result"] = result
    except Exception as exc:
        _train_status["status"] = f"failed: {exc}"


@router.post(
    "/train",
    response_model=ModelTrainResponse,
    summary="Train the Random Forest risk classifier",
    description=(
        "Reads all ground truth JSON sidecars from the dataset, extracts the 11-feature "
        "vector (vitals + trends + POD + red flags), and trains a balanced Random Forest. "
        "Runs in the background; check GET /model/status after a few seconds."
    ),
)
def train(background_tasks: BackgroundTasks):
    if _train_status.get("status") == "running":
        raise HTTPException(status_code=409, detail="Model training already in progress.")

    background_tasks.add_task(_do_train)

    return ModelTrainResponse(
        status="queued",
        model_path=settings.rf_path,
        message="Model training started in background. Check GET /model/status for completion.",
    )
