"""
routers/analyze.py

POST /analyze          — Run pipeline on a single uploaded PDF
POST /analyze/batch    — Queue a batch of PDFs for async processing
GET  /jobs/{job_id}/status  — Poll batch job status
GET  /jobs/{job_id}/results — Retrieve batch job results
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, BackgroundTasks

from app.core.pipeline import run_single_pdf, run_pdf_list
from app.schemas.models import (
    AnalyzeResponse,
    BatchJobResponse,
    BatchJobStatus,
    BatchResultsResponse,
    BatchResultRow,
)
from app.config import get_settings

router = APIRouter()
settings = get_settings()

# ── In-memory job store (replace with Redis/DB in production) ──────────────
_jobs: Dict[str, Dict[str, Any]] = {}


# ─────────────────────────────────────────────────────────────────────────────
# POST /analyze — single PDF
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "",
    response_model=AnalyzeResponse,
    summary="Analyze a single post-op PDF",
    description=(
        "Upload a post-op patient PDF. The pipeline will extract clinical features, "
        "retrieve similar cases and guidelines from FAISS, perform temporal analysis, "
        "and return a structured risk assessment."
    ),
)
async def analyze_single(
    file: UploadFile = File(..., description="Post-op patient PDF"),
    top_k: int = Form(default=5, description="Number of RAG results to retrieve"),
    use_ollama: bool = Form(default=False, description="Enable local Ollama LLM reasoning"),
    debug: bool = Form(default=False, description="Include trace and debug info in response"),
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        result = run_single_pdf(
            pdf_bytes=pdf_bytes,
            pdf_filename=file.filename,
            dataset_root=settings.dataset_root,
            faiss_cases_dir=settings.faiss_cases_dir,
            faiss_guidelines_dir=settings.faiss_guidelines_dir,
            top_k=top_k,
            use_ollama=use_ollama,
            rf_path=settings.rf_path,
            debug=debug,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {exc}")

    return AnalyzeResponse(**result)


# ─────────────────────────────────────────────────────────────────────────────
# POST /analyze/batch — multiple PDFs, async background processing
# ─────────────────────────────────────────────────────────────────────────────

def _run_batch_job(job_id: str, pdf_items: List[Dict[str, Any]], kwargs: Dict[str, Any]):
    """Background task: processes each PDF and stores results in _jobs."""
    _jobs[job_id]["status"] = "running"
    results = []
    failed = 0

    for item in pdf_items:
        try:
            res = run_single_pdf(
                pdf_bytes=item["bytes"],
                pdf_filename=item["filename"],
                **kwargs,
            )
            results.append(res)
            _jobs[job_id]["processed"] += 1
        except Exception as exc:
            failed += 1
            _jobs[job_id]["processed"] += 1
            results.append({
                "pdf_name": item["filename"],
                "risk_assessment": {"risk_level": None, "risk_score": None, "alerts": []},
                "errors": [str(exc)],
            })

    _jobs[job_id]["status"] = "done"
    _jobs[job_id]["failed"] = failed
    _jobs[job_id]["results"] = results


@router.post(
    "/batch",
    response_model=BatchJobResponse,
    summary="Analyze multiple PDFs asynchronously",
    description=(
        "Upload multiple post-op PDFs. The job runs in the background. "
        "Poll GET /jobs/{job_id}/status to check progress, "
        "then GET /jobs/{job_id}/results to retrieve all results."
    ),
)
async def analyze_batch(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="One or more post-op PDFs"),
    top_k: int = Form(default=5),
    use_ollama: bool = Form(default=False),
):
    pdf_items = []
    for f in files:
        if not f.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"{f.filename} is not a PDF.")
        pdf_items.append({"filename": f.filename, "bytes": await f.read()})

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "status": "queued",
        "total_files": len(pdf_items),
        "processed": 0,
        "failed": 0,
        "results": [],
    }

    kwargs = dict(
        dataset_root=settings.dataset_root,
        faiss_cases_dir=settings.faiss_cases_dir,
        faiss_guidelines_dir=settings.faiss_guidelines_dir,
        top_k=top_k,
        use_ollama=use_ollama,
        rf_path=settings.rf_path,
        debug=False,
    )

    background_tasks.add_task(_run_batch_job, job_id, pdf_items, kwargs)

    return BatchJobResponse(
        job_id=job_id,
        status="queued",
        total_files=len(pdf_items),
        message=f"Batch job queued with {len(pdf_items)} file(s). Poll /jobs/{job_id}/status for progress.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# GET /jobs/{job_id}/status
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/jobs/{job_id}/status",
    response_model=BatchJobStatus,
    summary="Poll batch job status",
)
def job_status(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
    return BatchJobStatus(
        job_id=job_id,
        status=job["status"],
        total_files=job["total_files"],
        processed=job["processed"],
        failed=job["failed"],
        results_available=job["status"] == "done",
    )


# ─────────────────────────────────────────────────────────────────────────────
# GET /jobs/{job_id}/results
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/jobs/{job_id}/results",
    response_model=BatchResultsResponse,
    summary="Retrieve batch job results",
)
def job_results(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
    if job["status"] != "done":
        raise HTTPException(
            status_code=202,
            detail=f"Job is still {job['status']}. Check /jobs/{job_id}/status first.",
        )

    results = job.get("results", [])
    tier_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
    rows = []

    for r in results:
        risk = r.get("risk_assessment") or {}
        level = risk.get("risk_level")
        if level in tier_counts:
            tier_counts[level] += 1

        rows.append(BatchResultRow(
            pdf=r.get("pdf_name", ""),
            patient_id=r.get("patient_id"),
            pod=r.get("pod"),
            procedure=r.get("procedure"),
            risk_level=level,
            risk_score=risk.get("risk_score"),
            alerts=risk.get("alerts") or [],
            citations_similar_cases=(r.get("retrieved_context") or {}).get("similar_cases_citations") or [],
            citations_guidelines=(r.get("retrieved_context") or {}).get("guidelines_citations") or [],
            parser_quality=r.get("parser_quality"),
            errors=r.get("errors") or [],
        ))

    return BatchResultsResponse(
        job_id=job_id,
        status=job["status"],
        total_files=job["total_files"],
        risk_tier_counts=tier_counts,
        results=rows,
    )
