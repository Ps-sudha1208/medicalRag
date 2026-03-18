"""
schemas/models.py — All Pydantic request and response models for the API.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# Shared / reused building blocks
# ─────────────────────────────────────────────

class ValidationResult(BaseModel):
    status: str = Field(..., description="'ok', 'warn', or 'error'")
    errors: List[str] = Field(default_factory=list, description="Out-of-range physiological values")
    warnings: List[str] = Field(default_factory=list, description="Missing required fields")


class TemporalAnalysis(BaseModel):
    status: str
    history_points_used: int
    temperature_trend: Optional[str] = None
    heart_rate_trend: Optional[str] = None
    wbc_trend: Optional[str] = None
    spo2_trend: Optional[str] = None
    expected_ranges_bucket: Optional[str] = None
    deviations: List[str] = Field(default_factory=list)
    rapid_deterioration: Optional[Dict[str, bool]] = None
    rapid_deterioration_any: bool = False


class RiskAssessment(BaseModel):
    risk_level: str = Field(..., description="LOW | MEDIUM | HIGH | CRITICAL")
    risk_score: int
    alerts: List[str] = Field(default_factory=list)
    rule_score: int
    rule_level: str
    rule_factors: List[str] = Field(default_factory=list)
    ml_pred: Optional[str] = None
    ml_proba: Optional[Dict[str, float]] = None
    final_factors: List[str] = Field(default_factory=list)
    llm_reasoning: Optional[str] = None


class RetrievedContext(BaseModel):
    query_used: Optional[str] = None
    similar_cases_citations: List[str] = Field(default_factory=list)
    guidelines_citations: List[str] = Field(default_factory=list)


# ─────────────────────────────────────────────
# /analyze — single PDF response
# ─────────────────────────────────────────────

class AnalyzeResponse(BaseModel):
    pdf_name: str
    patient_id: Optional[str] = None
    pod: Optional[int] = None
    procedure: Optional[str] = None
    parser_quality: Optional[float] = Field(None, description="0.0–1.0 fraction of required fields extracted")
    validation: ValidationResult
    temporal_analysis: TemporalAnalysis
    risk_assessment: RiskAssessment
    retrieved_context: RetrievedContext
    final_response: str
    errors: List[str] = Field(default_factory=list)
    trace: List[str] = Field(default_factory=list)


# ─────────────────────────────────────────────
# /analyze/batch — batch job responses
# ─────────────────────────────────────────────

class BatchJobResponse(BaseModel):
    job_id: str
    status: str = Field(..., description="'queued' | 'running' | 'done' | 'failed'")
    total_files: int
    message: str


class BatchJobStatus(BaseModel):
    job_id: str
    status: str
    total_files: int
    processed: int
    failed: int
    results_available: bool


class BatchResultRow(BaseModel):
    pdf: str
    patient_id: Optional[str] = None
    pod: Optional[int] = None
    procedure: Optional[str] = None
    risk_level: Optional[str] = None
    risk_score: Optional[int] = None
    alerts: List[str] = Field(default_factory=list)
    citations_similar_cases: List[str] = Field(default_factory=list)
    citations_guidelines: List[str] = Field(default_factory=list)
    parser_quality: Optional[float] = None
    errors: List[str] = Field(default_factory=list)


class BatchResultsResponse(BaseModel):
    job_id: str
    status: str
    total_files: int
    risk_tier_counts: Dict[str, int]
    results: List[BatchResultRow]


# ─────────────────────────────────────────────
# /index
# ─────────────────────────────────────────────

class IndexRebuildResponse(BaseModel):
    status: str
    index_type: str = Field(..., description="'cases' or 'guidelines'")
    chunks_built: Optional[int] = None
    vectordb_dir: Optional[str] = None
    message: str


class IndexStatusResponse(BaseModel):
    cases_index_exists: bool
    cases_index_dir: str
    guidelines_index_exists: bool
    guidelines_index_dir: str
    embedding_model: str
    embedding_dim: int


# ─────────────────────────────────────────────
# /model
# ─────────────────────────────────────────────

class ModelTrainResponse(BaseModel):
    status: str
    model_path: str
    samples_trained_on: Optional[int] = None
    message: str


class ModelStatusResponse(BaseModel):
    model_exists: bool
    model_path: str
    message: str


# ─────────────────────────────────────────────
# /evaluate
# ─────────────────────────────────────────────

class PerClassMetrics(BaseModel):
    precision: float
    recall: float
    f1: float
    support: int


class EvaluationResponse(BaseModel):
    matched_samples: int
    skipped_missing_gt: int
    accuracy: float
    critical_recall: float
    alert_rate: float
    per_tier: Dict[str, PerClassMetrics]
    confusion_matrix: Dict[str, Dict[str, int]]
    alert_rate_by_tier: Dict[str, Dict[str, Any]]


# ─────────────────────────────────────────────
# /patients
# ─────────────────────────────────────────────

class DailyObservationSummary(BaseModel):
    pod: Optional[int] = None
    timestamp: Optional[str] = None
    temperature: Optional[float] = None
    heart_rate: Optional[int] = None
    blood_pressure: Optional[str] = None
    spo2: Optional[int] = None
    wbc: Optional[float] = None
    pain_score: Optional[int] = None
    rule_risk_level: Optional[str] = None
    rule_risk_score: Optional[int] = None
    clinical_notes: Optional[str] = None
    injected_events: List[str] = Field(default_factory=list)


class PatientHistoryResponse(BaseModel):
    patient_id: str
    total_observations: int
    history: List[DailyObservationSummary]


class PatientPodResponse(BaseModel):
    patient_id: str
    pod: int
    daily_observation: DailyObservationSummary
    ground_truth: Optional[Dict[str, Any]] = None
    patient_profile: Optional[Dict[str, Any]] = None
