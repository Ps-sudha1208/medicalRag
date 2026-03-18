"""
main.py — FastAPI entrypoint for the Clinical LangGraph Pipeline.

Run with:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

Swagger UI available at:
    http://localhost:8000/docs
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import analyze, index, model, evaluate, patients

app = FastAPI(
    title="Clinical Risk Pipeline API",
    description=(
        "Multi-agent post-operative risk assessment pipeline. "
        "Accepts patient PDFs, extracts clinical features, retrieves similar cases "
        "and guidelines via FAISS RAG, performs temporal analysis, and returns a "
        "structured risk tier (LOW / MEDIUM / HIGH / CRITICAL) with alerts and citations."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze.router,   prefix="/analyze",   tags=["Analyze"])
app.include_router(index.router,     prefix="/index",     tags=["Index Management"])
app.include_router(model.router,     prefix="/model",     tags=["ML Model"])
app.include_router(evaluate.router,  prefix="/evaluate",  tags=["Evaluation"])
app.include_router(patients.router,  prefix="/patients",  tags=["Patients"])


@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "Clinical Risk Pipeline API is running."}


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok"}
