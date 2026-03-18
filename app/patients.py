"""
routers/patients.py

GET /patients                           — List all patient IDs
GET /patients/{patient_id}/history      — Full longitudinal history for a patient
GET /patients/{patient_id}/pod/{pod}    — Single POD record for a patient
"""

from typing import List

from fastapi import APIRouter, HTTPException

from app.core.patients import get_patient_history, get_patient_pod_record, list_all_patients
from app.schemas.models import PatientHistoryResponse, PatientPodResponse, DailyObservationSummary
from app.config import get_settings

router = APIRouter()
settings = get_settings()


# ─────────────────────────────────────────────────────────────────────────────
# GET /patients
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "",
    response_model=List[str],
    summary="List all patient IDs",
    description="Returns a sorted list of all patient IDs found in the dataset jsons/ directory.",
)
def list_patients():
    try:
        return list_all_patients(settings.dataset_root)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# GET /patients/{patient_id}/history
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/{patient_id}/history",
    response_model=PatientHistoryResponse,
    summary="Get longitudinal history for a patient",
    description=(
        "Returns all daily observation records for a given patient, "
        "sorted by Post-Op Day (POD) ascending."
    ),
)
def patient_history(patient_id: str):
    try:
        observations = get_patient_history(settings.dataset_root, patient_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    if not observations:
        raise HTTPException(status_code=404, detail=f"No records found for patient {patient_id}.")

    history = []
    for obs in observations:
        history.append(DailyObservationSummary(
            pod=obs.get("pod"),
            timestamp=obs.get("timestamp"),
            temperature=obs.get("temperature"),
            heart_rate=obs.get("heart_rate"),
            blood_pressure=obs.get("blood_pressure"),
            spo2=obs.get("spo2"),
            wbc=obs.get("wbc"),
            pain_score=obs.get("pain_score"),
            rule_risk_level=obs.get("rule_risk_level"),
            rule_risk_score=obs.get("rule_risk_score"),
            clinical_notes=obs.get("clinical_notes"),
            injected_events=obs.get("injected_events") or [],
        ))

    return PatientHistoryResponse(
        patient_id=patient_id,
        total_observations=len(history),
        history=history,
    )


# ─────────────────────────────────────────────────────────────────────────────
# GET /patients/{patient_id}/pod/{pod}
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/{patient_id}/pod/{pod}",
    response_model=PatientPodResponse,
    summary="Get a single POD record for a patient",
    description=(
        "Returns the full JSON sidecar for a specific patient and Post-Op Day, "
        "including ground truth risk level, patient profile, and daily observation."
    ),
)
def patient_pod(patient_id: str, pod: int):
    try:
        record = get_patient_pod_record(settings.dataset_root, patient_id, pod)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    if record is None:
        raise HTTPException(
            status_code=404,
            detail=f"No record found for patient {patient_id} at POD {pod}.",
        )

    dob = record.get("daily_observation", {})
    obs = DailyObservationSummary(
        pod=dob.get("pod"),
        timestamp=dob.get("timestamp"),
        temperature=dob.get("temperature"),
        heart_rate=dob.get("heart_rate"),
        blood_pressure=dob.get("blood_pressure"),
        spo2=dob.get("spo2"),
        wbc=dob.get("wbc"),
        pain_score=dob.get("pain_score"),
        rule_risk_level=dob.get("rule_risk_level"),
        rule_risk_score=dob.get("rule_risk_score"),
        clinical_notes=dob.get("clinical_notes"),
        injected_events=dob.get("injected_events") or [],
    )

    return PatientPodResponse(
        patient_id=patient_id,
        pod=pod,
        daily_observation=obs,
        ground_truth=record.get("ground_truth"),
        patient_profile=record.get("patient_profile"),
    )
