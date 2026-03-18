"""
routers/index.py

POST /index/cases/rebuild       — Rebuild the patient-cases FAISS index
POST /index/guidelines/rebuild  — Rebuild the guidelines FAISS index
GET  /index/status              — Check whether both indexes exist
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks

from app.core.indexer import rebuild_cases_index, rebuild_guidelines_index, get_index_status
from app.schemas.models import IndexRebuildResponse, IndexStatusResponse
from app.config import get_settings

router = APIRouter()
settings = get_settings()

# ── Rebuild status tracker (simple in-memory, sufficient for single-server) ─
_rebuild_status: dict = {}


# ─────────────────────────────────────────────────────────────────────────────
# GET /index/status
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/status",
    response_model=IndexStatusResponse,
    summary="Check FAISS index status",
    description="Returns whether the cases and guidelines indexes exist on disk.",
)
def index_status():
    try:
        return get_index_status(
            faiss_cases_dir=settings.faiss_cases_dir,
            faiss_guidelines_dir=settings.faiss_guidelines_dir,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# POST /index/cases/rebuild
# ─────────────────────────────────────────────────────────────────────────────

def _do_rebuild_cases():
    _rebuild_status["cases"] = "running"
    try:
        result = rebuild_cases_index(
            dataset_root=settings.dataset_root,
            faiss_cases_dir=settings.faiss_cases_dir,
        )
        _rebuild_status["cases"] = "done"
        _rebuild_status["cases_result"] = result
    except Exception as exc:
        _rebuild_status["cases"] = f"failed: {exc}"


@router.post(
    "/cases/rebuild",
    response_model=IndexRebuildResponse,
    summary="Rebuild patient-cases FAISS index",
    description=(
        "Scans all PDFs in the dataset, chunks and embeds them with MiniLM-L6-v2, "
        "and saves a fresh FAISS index. Runs in the background; check /index/status for completion."
    ),
)
def rebuild_cases(background_tasks: BackgroundTasks):
    if _rebuild_status.get("cases") == "running":
        raise HTTPException(status_code=409, detail="Cases index rebuild already in progress.")

    background_tasks.add_task(_do_rebuild_cases)
    return IndexRebuildResponse(
        status="queued",
        index_type="cases",
        message="Cases index rebuild started in background.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# POST /index/guidelines/rebuild
# ─────────────────────────────────────────────────────────────────────────────

def _do_rebuild_guidelines():
    _rebuild_status["guidelines"] = "running"
    try:
        result = rebuild_guidelines_index(
            guidelines_dir=settings.guidelines_dir,
            faiss_guidelines_dir=settings.faiss_guidelines_dir,
        )
        _rebuild_status["guidelines"] = "done"
        _rebuild_status["guidelines_result"] = result
    except Exception as exc:
        _rebuild_status["guidelines"] = f"failed: {exc}"


@router.post(
    "/guidelines/rebuild",
    response_model=IndexRebuildResponse,
    summary="Rebuild clinical guidelines FAISS index",
    description=(
        "Loads all PDFs/TXT/MD files from the guidelines directory, "
        "chunks and embeds them, and saves a fresh FAISS index."
    ),
)
def rebuild_guidelines(background_tasks: BackgroundTasks):
    if _rebuild_status.get("guidelines") == "running":
        raise HTTPException(status_code=409, detail="Guidelines index rebuild already in progress.")

    background_tasks.add_task(_do_rebuild_guidelines)
    return IndexRebuildResponse(
        status="queued",
        index_type="guidelines",
        message="Guidelines index rebuild started in background.",
    )
