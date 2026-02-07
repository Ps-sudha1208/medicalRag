"""
generate_synthetic_dataset.py

Writes ONLY here (no clutter):
  Data/synthetic_dataset/
    pdfs/
    jsons/
    manifest.json

Synthetic Post-Op PDF Generator (for Multi-Agent RAG / LangGraph)

Creates:
- Synthetic post-op clinical PDFs (messy, semi-structured text + tables)
- Ground-truth JSON sidecars for evaluation (features, events, risk label)

Output directory structure:
./synthetic_postop_pdfs/
  PT0001_POD1_LOW_20260205.pdf
  PT0001_POD1_LOW_20260205.json
  ...

What’s updated per your needs:
1) Consolidated outputs into separate folders:
   - <output_root>/pdfs/        (all PDFs)
   - <output_root>/jsons/       (all JSON ground truth)
   - <output_root>/manifest.json

2) Split by PATIENTS (not PDFs) and assign each patient a schedule type:
   - dense_daily_30 : POD 0..30 (31 PDFs/patient)
   - dense_early_0_5: POD 0..5  (6 PDFs/patient)
   - gapped_30      : checkpoints over 30 days (8 PDFs/patient by default)

Default patient split ratios (can be changed):
  dense_daily_30 : 30%
  For each patient in this group: we get a pdf every day from POD(post-op day)0 to POD 30 
  dense_early_0_5: 10%
  For each patient in this group: we get a pdf for the days: 0, 1, 2, 3, 4, 5
  gapped_30      : 60%
  For each patient in this group: we get a pdf in the gaps: 0, 1, 3, 5, 7, 14, 21, 30
Dependencies:
  pip install reportlab faker numpy
"""

import argparse
import json
import random
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from faker import Faker

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet


# -----------------------------
# Helpers
# -----------------------------
def project_root_from_cwd() -> Path:
    """
    Assumes you run from repo root or inside repo.
    Finds a folder containing Data/.
    """
    cwd = Path.cwd().resolve()
    if (cwd / "Data").exists():
        return cwd
    if (cwd.parent / "Data").exists():
        return cwd.parent
    # fallback: create Data in cwd
    (cwd / "Data").mkdir(parents=True, exist_ok=True)
    return cwd


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


# -----------------------------
# Config
# -----------------------------
fake = Faker("en_IN")

SURGERIES = [
    ("Laparoscopic Cholecystectomy", "LOW"),
    ("Appendectomy", "LOW"),
    ("Total Hip Arthroplasty", "MED"),
    ("CABG (Coronary Artery Bypass Graft)", "HIGH"),
    ("Whipple Procedure (Pancreaticoduodenectomy)", "HIGH"),
    ("Colectomy (Partial)", "MED"),
    ("Hernia Repair (Inguinal)", "LOW"),
    ("Total Abdominal Hysterectomy", "MED"),
]

COMORBIDITIES = [
    "Diabetes mellitus", "Hypertension", "COPD", "CKD stage 3",
    "Coronary artery disease", "Obesity", "Smoker", "None"
]

COMPLICATIONS = [
    ("Surgical site infection", (2, 7), "MED"),
    ("Pneumonia", (2, 10), "MED"),
    ("Pulmonary embolism", (1, 10), "CRIT"),
    ("Sepsis", (1, 14), "CRIT"),
    ("Hemorrhage", (0, 3), "CRIT"),
    ("Wound dehiscence", (3, 21), "HIGH"),
    ("Atelectasis", (0, 5), "LOW"),
    ("UTI", (2, 21), "LOW"),
]

RISK_LEVELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

DEFAULT_SCHEDULES = {
    "dense_daily_30": list(range(0, 31)),              # POD 0..30 => 31 docs
    "dense_early_0_5": list(range(0, 6)),              # POD 0..5  => 6 docs
    "gapped_30": [0, 1, 3, 5, 7, 14, 21, 30],          # 8 docs
}

DEFAULT_SPLIT_RATIOS = {
    "gapped_30": 0.60,
    "dense_daily_30": 0.30,
    "dense_early_0_5": 0.10,
}


# -----------------------------
# Data models
# -----------------------------
@dataclass
class PatientProfile:
    patient_id: str
    name: str
    age: int
    sex: str
    comorbidities: List[str]
    surgery: str
    surgery_complexity: str
    surgeon: str
    schedule_type: str
    pods: List[int]


@dataclass
class DailyObservation:
    pod: int
    timestamp: str
    temperature: float
    heart_rate: int
    blood_pressure: str
    respiratory_rate: int
    spo2: int
    pain_score: int
    wbc: float
    hemoglobin: float
    wound_status: str
    drainage: str
    clinical_notes: str
    injected_events: List[str]
    rule_risk_score: int
    rule_risk_level: str


# -----------------------------
# Synthetic generation logic
# -----------------------------
def choose_surgery() -> Tuple[str, str]:
    surgery, base = random.choice(SURGERIES)
    if base == "MED":
        return surgery, "MEDIUM"
    return surgery, base


def sample_comorbidities() -> List[str]:
    k = random.choice([0, 1, 2, 3])
    choices = random.sample(COMORBIDITIES[:-1], k=k) if k > 0 else ["None"]
    if "None" in choices and len(choices) > 1:
        choices = [c for c in choices if c != "None"]
    return choices


def baseline_vitals(age: int) -> Dict[str, float]:
    return {
        "temp": np.random.normal(37.0, 0.2),
        "hr": np.random.normal(82, 10),
        "rr": np.random.normal(16, 2),
        "spo2": np.random.normal(97, 1.5),
        "wbc": np.random.normal(8.0, 1.5),
        "hgb": np.random.normal(12.5, 1.2),
        "bps": np.random.normal(122, 12),
        "bpd": np.random.normal(78, 8),
        "pain": np.random.normal(6.0, 1.8),
    }


def add_postop_trend(pod: int, base: Dict[str, float]) -> Dict[str, float]:
    temp = base["temp"] + (0.3 * np.exp(-pod/2.0)) + np.random.normal(0, 0.15)
    hr = base["hr"] + (10 * np.exp(-pod/2.2)) + np.random.normal(0, 6)
    rr = base["rr"] + np.random.normal(0, 1.5)
    spo2 = base["spo2"] + np.random.normal(0, 1.0)
    wbc = base["wbc"] + (2.0 * np.exp(-pod/2.5)) + np.random.normal(0, 0.8)
    hgb = base["hgb"] - (0.3 * np.exp(-pod/3.0)) + np.random.normal(0, 0.4)
    bps = base["bps"] + np.random.normal(0, 8)
    bpd = base["bpd"] + np.random.normal(0, 6)
    pain = base["pain"] - (1.0 * (pod/2.5)) + np.random.normal(0, 0.9)

    return {
        "temp": float(temp),
        "hr": float(hr),
        "rr": float(rr),
        "spo2": float(spo2),
        "wbc": float(wbc),
        "hgb": float(hgb),
        "bps": float(bps),
        "bpd": float(bpd),
        "pain": float(pain),
    }


def inject_complications(
    pod: int,
    values: Dict[str, float],
    comorbidities: List[str],
    planned_events: List[Tuple[str, int, str]]
) -> Tuple[Dict[str, float], List[str], str, str, str]:
    events_today = [e for e in planned_events if e[1] == pod]
    ongoing = [e for e in planned_events if e[1] <= pod]

    wound_status = "Clean, dry, intact"
    drainage = "None"
    notes_bits: List[str] = []
    injected: List[str] = []

    if pod <= 2 and random.random() < 0.35:
        drainage = random.choice(["Serosanguinous small", "Serous scant"])
        wound_status = random.choice(["Mild erythema", "Edges approximated with mild erythema"])

    for name, onset, severity in ongoing:
        days_since = pod - onset
        ramp = 1.0 if days_since == 0 else 0.7 if days_since == 1 else 0.5

        if name == "Surgical site infection":
            values["temp"] += ramp * random.uniform(0.6, 1.6)
            values["wbc"] += ramp * random.uniform(2.5, 8.0)
            wound_status = random.choice(["Erythema + warmth", "Purulent discharge", "Increasing redness"])
            drainage = random.choice(["Purulent moderate", "Seropurulent"])
            notes_bits.append("Concern for SSI; wound erythematous with drainage.")
        elif name == "Pneumonia":
            values["temp"] += ramp * random.uniform(0.6, 1.4)
            values["rr"] += ramp * random.uniform(3, 8)
            values["spo2"] -= ramp * random.uniform(3, 10)
            notes_bits.append("Productive cough, crackles on auscultation; consider pneumonia.")
        elif name == "Pulmonary embolism":
            values["hr"] += ramp * random.uniform(20, 50)
            values["rr"] += ramp * random.uniform(6, 12)
            values["spo2"] -= ramp * random.uniform(8, 18)
            notes_bits.append("Sudden dyspnea/chest pain; rule out pulmonary embolism.")
        elif name == "Sepsis":
            values["temp"] += ramp * random.uniform(1.0, 2.5)
            values["hr"] += ramp * random.uniform(25, 60)
            values["wbc"] += ramp * random.uniform(4.0, 15.0)
            values["bps"] -= ramp * random.uniform(15, 40)
            notes_bits.append("Sepsis suspected; hypotension and tachycardia present.")
            notes_bits.append("RED FLAG: sepsis")
        elif name == "Hemorrhage":
            values["hr"] += ramp * random.uniform(20, 45)
            values["bps"] -= ramp * random.uniform(20, 55)
            values["hgb"] -= ramp * random.uniform(1.0, 3.0)
            notes_bits.append("Active bleeding suspected; monitor hemoglobin and hemodynamics.")
            notes_bits.append("RED FLAG: hemorrhage")
        elif name == "Wound dehiscence":
            wound_status = "Wound dehiscence noted"
            drainage = "Serosanguinous large"
            values["temp"] += ramp * random.uniform(0.2, 1.0)
            notes_bits.append("Wound edges separated; urgent surgical review recommended.")
            notes_bits.append("RED FLAG: wound dehiscence")
        elif name == "Atelectasis":
            values["spo2"] -= ramp * random.uniform(1, 5)
            notes_bits.append("Likely atelectasis; encourage incentive spirometry.")
        elif name == "UTI":
            values["temp"] += ramp * random.uniform(0.3, 1.0)
            values["wbc"] += ramp * random.uniform(1.5, 5.0)
            notes_bits.append("Dysuria/urinary frequency; consider UTI.")

    for e in events_today:
        injected.append(e[0])

    noise = []
    if random.random() < 0.30:
        noise.append(random.choice([
            "Pt c/o mild nausea.", "Ambulating with assistance.", "No SOB.", "Tol PO.", "D/c planning started."
        ]))
    if "Diabetes mellitus" in comorbidities and random.random() < 0.25:
        noise.append("BGL elevated; insulin sliding scale.")
    if "COPD" in comorbidities and random.random() < 0.25:
        noise.append("Wheezing noted; bronchodilator PRN.")

    clinical_notes = " ".join(notes_bits + noise).strip()
    if not clinical_notes:
        clinical_notes = random.choice([
            "Recovering as expected. Pain controlled. No acute distress.",
            "Stable post-op course. Vitals within expected limits.",
            "Mild pain, improving mobility. No concerning symptoms."
        ])

    return values, injected, wound_status, drainage, clinical_notes


def rule_based_risk_score(features: Dict) -> Tuple[int, str]:
    score = 0
    temp = features.get("temperature", 0)
    hr = features.get("heart_rate", 0)
    bps = features.get("bp_systolic", 120)
    spo2 = features.get("spo2", 100)
    wbc = features.get("wbc", 7)
    pain = features.get("pain_score", 0)
    notes = features.get("clinical_notes", "").lower()

    if temp > 39.0:
        score += 30
    elif temp > 38.0:
        score += 15

    if hr > 120:
        score += 25
    elif hr > 100:
        score += 15

    if bps < 90:
        score += 25

    if spo2 < 90:
        score += 30
    elif spo2 < 95:
        score += 15

    if wbc > 15.0:
        score += 20
    elif wbc > 11.0:
        score += 10

    if pain > 7:
        score += 15
    elif pain > 5:
        score += 5

    for flag in ["sepsis", "wound dehiscence", "pulmonary embolism", "hemorrhage", "shock"]:
        if flag in notes:
            score += 40

    if score >= 76:
        level = "CRITICAL"
    elif score >= 51:
        level = "HIGH"
    elif score >= 26:
        level = "MEDIUM"
    else:
        level = "LOW"
    return int(score), level


def plan_patient_events(base_risk: str, max_pod: int) -> List[Tuple[str, int, str]]:
    events: List[Tuple[str, int, str]] = []
    p_any = {"LOW": 0.15, "MEDIUM": 0.30, "HIGH": 0.55, "CRITICAL": 0.75}[base_risk]
    if random.random() > p_any:
        return events

    n_events = 1 if random.random() < 0.75 else 2

    for _ in range(n_events):
        name, (lo, hi), sev = random.choice(COMPLICATIONS)
        onset = random.randint(lo, min(hi, max_pod))
        events.append((name, onset, sev))

    if base_risk == "CRITICAL" and not any(e[2] == "CRIT" for e in events):
        name, (lo, hi), sev = random.choice([c for c in COMPLICATIONS if c[2] == "CRIT"])
        onset = random.randint(lo, min(hi, max_pod))
        events.append((name, onset, sev))

    dedup = {}
    for e in events:
        if e[0] not in dedup or e[1] < dedup[e[0]][1]:
            dedup[e[0]] = e
    return list(dedup.values())


def make_profile(idx: int, schedule_type: str, pods: List[int]) -> PatientProfile:
    surgery, _ = choose_surgery()
    age = random.randint(18, 85)
    sex = random.choice(["F", "M"])
    comb = sample_comorbidities()
    if comb == ["None"] and random.random() < 0.25:
        comb = [random.choice(COMORBIDITIES[:-1])]

    complexity = random.choice(["low", "moderate", "high"])
    surgeon = f"Dr. {fake.last_name()}"

    return PatientProfile(
        patient_id=f"PT{idx:04d}",
        name=fake.name(),
        age=age,
        sex=sex,
        comorbidities=comb,
        surgery=surgery,
        surgery_complexity=complexity,
        surgeon=surgeon,
        schedule_type=schedule_type,
        pods=pods,
    )


def generate_daily_obs(
    profile: PatientProfile,
    pod: int,
    surgery_date: datetime,
    base_vals: Dict[str, float],
    planned_events: List[Tuple[str, int, str]],
) -> DailyObservation:
    ts = surgery_date + timedelta(days=pod, hours=random.randint(6, 10), minutes=random.choice([0, 15, 30, 45]))
    vals = add_postop_trend(pod, base_vals)

    vals, injected, wound_status, drainage, clinical_notes = inject_complications(
        pod=pod,
        values=vals,
        comorbidities=profile.comorbidities,
        planned_events=planned_events
    )

    temp = clamp(vals["temp"], 35.0, 42.0)
    hr = int(clamp(vals["hr"], 40, 200))
    rr = int(clamp(vals["rr"], 8, 40))
    spo2 = int(clamp(vals["spo2"], 70, 100))
    wbc = round(float(clamp(vals["wbc"], 1.0, 50.0)), 1)
    hgb = round(float(clamp(vals["hgb"], 4.0, 18.0)), 1)
    bps = int(clamp(vals["bps"], 60, 220))
    bpd = int(clamp(vals["bpd"], 30, 140))
    pain = int(clamp(vals["pain"], 0, 10))

    bp_str = f"{bps}/{bpd}"

    features = {
        "temperature": float(f"{temp:.1f}"),
        "heart_rate": hr,
        "bp_systolic": bps,
        "bp_diastolic": bpd,
        "spo2": spo2,
        "wbc": wbc,
        "pain_score": pain,
        "clinical_notes": clinical_notes,
    }
    rule_score, rule_level = rule_based_risk_score(features)

    return DailyObservation(
        pod=pod,
        timestamp=ts.isoformat(),
        temperature=float(f"{temp:.1f}"),
        heart_rate=hr,
        blood_pressure=bp_str,
        respiratory_rate=rr,
        spo2=spo2,
        pain_score=pain,
        wbc=wbc,
        hemoglobin=hgb,
        wound_status=wound_status,
        drainage=drainage,
        clinical_notes=clinical_notes,
        injected_events=injected,
        rule_risk_score=rule_score,
        rule_risk_level=rule_level,
    )


# -----------------------------
# PDF Rendering
# -----------------------------
def build_pdf(out_pdf_path: Path, profile: PatientProfile, obs: DailyObservation, surgery_date: datetime):
    styles = getSampleStyleSheet()
    normal = styles["Normal"]
    title = styles["Title"]
    h2 = styles["Heading2"]
    h3 = styles["Heading3"]

    doc = SimpleDocTemplate(
        str(out_pdf_path),
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        title=f"Post-Op Note {profile.patient_id} POD{obs.pod}",
    )

    story = []
    story.append(Paragraph("Post-Operative Progress Note", title))
    story.append(Spacer(1, 0.15 * inch))

    header_lines = [
        f"<b>Patient ID:</b> {profile.patient_id}",
        f"<b>Name:</b> {profile.name}",
        f"<b>Age/Sex:</b> {profile.age}/{profile.sex}",
        f"<b>Procedure:</b> {profile.surgery}",
        f"<b>Surgeon:</b> {profile.surgeon}",
        f"<b>Surgery Date:</b> {surgery_date.strftime('%Y-%m-%d')}",
        f"<b>Post-Op Day</b> {obs.pod}",
        f"<b>Timestamp:</b> {obs.timestamp}",
        f"<b>Schedule Type:</b> {profile.schedule_type}",
    ]
    story.append(Paragraph("<br/>".join(header_lines), normal))
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("Vital Signs", h2))
    vitals_data = [
        ["Temperature", f"{obs.temperature:.1f}"],
        ["Heart Rate", f"{obs.heart_rate}"],
        ["Blood Pressure", f"{obs.blood_pressure}"],
        ["Respiratory Rate", f"{obs.respiratory_rate}"],
        ["SpO2", f"{obs.spo2}"],
        ["Pain Score", f"{obs.pain_score}"],
    ]
    t = Table(vitals_data, colWidths=[2.0 * inch, 3.0 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("Laboratory Values", h2))
    labs_data = [
        ["WBC", f"{obs.wbc:.1f}"],
        ["Hemoglobin", f"{obs.hemoglobin:.1f}"],
    ]
    lt = Table(labs_data, colWidths=[2.0 * inch, 3.0 * inch])
    lt.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
    ]))
    story.append(lt)
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("Wound / Drainage", h2))
    story.append(Paragraph(
        f"<b>Wound Status:</b> {obs.wound_status}<br/>"
        f"<b>Drainage:</b> {obs.drainage}",
        normal
    ))
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("Past Medical History", h2))
    story.append(Paragraph(", ".join(profile.comorbidities), normal))
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("Clinical Notes", h2))
    narrative = obs.clinical_notes
    if random.random() < 0.35:
        narrative += " " + random.choice([
            "Plan: cont abx? monitor.", "r/v labs am.", "if worse -> ER.",
            "Sx: fever? chills? denies.", "cxr prn."
        ])
    story.append(Paragraph(narrative, normal))
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("Automated Risk (Label)", h3))
    story.append(Paragraph(
        f"<b>Rule Risk Score:</b> {obs.rule_risk_score} / 100<br/>"
        f"<b>Rule Risk Level:</b> {obs.rule_risk_level}",
        normal
    ))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph(
        "<i>Disclaimer: This is synthetic, computer-generated content for testing only.</i>",
        styles["Italic"]
    ))

    doc.build(story)


# -----------------------------
# Schedule assignment
# -----------------------------
def allocate_schedule_counts(n_patients: int, ratios: Dict[str, float]) -> Dict[str, int]:
    keys = list(ratios.keys())
    raw = {k: n_patients * ratios[k] for k in keys}
    counts = {k: int(raw[k]) for k in keys}
    assigned = sum(counts.values())
    remainder = n_patients - assigned

    frac_order = sorted(keys, key=lambda k: (raw[k] - counts[k]), reverse=True)
    for i in range(remainder):
        counts[frac_order[i % len(frac_order)]] += 1
    return counts


def assign_schedule_types(n_patients: int, ratios: Dict[str, float], seed: int) -> List[str]:
    counts = allocate_schedule_counts(n_patients, ratios)
    schedule_list: List[str] = []
    for k, c in counts.items():
        schedule_list.extend([k] * c)
    rng = random.Random(seed + 999)
    rng.shuffle(schedule_list)
    return schedule_list


# -----------------------------
# Main dataset builder
# -----------------------------
def generate_dataset(
    output_root: Path,
    n_patients: int = 25,
    seed: int = 123,
    schedule_ratios: Optional[Dict[str, float]] = None,
    schedules: Optional[Dict[str, List[int]]] = None,
):
    random.seed(seed)
    np.random.seed(seed)

    schedule_ratios = schedule_ratios or DEFAULT_SPLIT_RATIOS
    schedules = schedules or DEFAULT_SCHEDULES

    out_root = Path(output_root)
    pdf_dir = out_root / "pdfs"
    json_dir = out_root / "jsons"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    # fixed for reproducibility
    anchor_today = datetime(2026, 2, 5)
    schedule_types = assign_schedule_types(n_patients, schedule_ratios, seed)

    manifest = {
        "generated_at": datetime.now().isoformat(),
        "output_root": str(out_root.resolve()),
        "pdf_dir": str(pdf_dir.resolve()),
        "json_dir": str(json_dir.resolve()),
        "n_patients": n_patients,
        "seed": seed,
        "schedule_ratios": schedule_ratios,
        "schedules": schedules,
        "patients": [],
        "files": [],
    }

    for i in range(1, n_patients + 1):
        schedule_type = schedule_types[i - 1]
        pods = schedules[schedule_type]
        max_pod = max(pods)

        profile = make_profile(i, schedule_type=schedule_type, pods=pods)

        # base risk prior influenced by comorbidities
        _, base_risk = choose_surgery()
        if any(c in profile.comorbidities for c in ["COPD", "CKD stage 3", "Coronary artery disease"]):
            base_risk = random.choice(["MEDIUM", "HIGH"])

        base_vals = baseline_vitals(profile.age)
        surgery_date = anchor_today - timedelta(days=random.randint(0, 3))
        planned_events = plan_patient_events(base_risk=base_risk, max_pod=max_pod)

        manifest["patients"].append({
            "patient_id": profile.patient_id,
            "schedule_type": schedule_type,
            "pods": pods,
            "planned_events": planned_events,
            "surgery_date": surgery_date.strftime("%Y-%m-%d"),
        })

        for pod in pods:
            obs = generate_daily_obs(profile, pod, surgery_date, base_vals, planned_events)

            date_tag = (anchor_today + timedelta(days=random.randint(0, 2))).strftime("%Y%m%d")
            base_name = f"{profile.patient_id}_POD{pod}_{obs.rule_risk_level}_{date_tag}"
            pdf_name = f"{base_name}.pdf"
            json_name = f"{base_name}.json"

            pdf_path = pdf_dir / pdf_name
            json_path = json_dir / json_name

            build_pdf(pdf_path, profile, obs, surgery_date)

            payload = {
                "patient_profile": asdict(profile),
                "daily_observation": asdict(obs),
                "planned_events": planned_events,
                "ground_truth": {
                    "risk_level_rule": obs.rule_risk_level,
                    "risk_score_rule": obs.rule_risk_score,
                    "schedule_type": schedule_type,
                    "has_complication": len([e for e in planned_events if e[1] <= pod]) > 0,
                    "complications_active": [e[0] for e in planned_events if e[1] <= pod],
                    "complications_started_today": obs.injected_events,
                }
            }

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)

            manifest["files"].append({
                "pdf": str(pdf_path.name),
                "json": str(json_path.name),
                "patient_id": profile.patient_id,
                "pod": pod,
                "risk_level": obs.rule_risk_level,
                "schedule_type": schedule_type,
            })

    manifest_path = out_root / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    schedule_counts = {}
    for p in manifest["patients"]:
        schedule_counts[p["schedule_type"]] = schedule_counts.get(p["schedule_type"], 0) + 1

    total_pdfs = len(manifest["files"])
    print(f"\n Output root: {out_root.resolve()}")
    print(f"PDFs:        {pdf_dir.resolve()}  ({total_pdfs} files)")
    print(f"JSONs:       {json_dir.resolve()} ({total_pdfs} files)")
    print(f"Manifest:    {manifest_path.resolve()}")
    print(f"Patient schedule counts: {schedule_counts}")
    print("Done.")


# -----------------------------
# CLI
# -----------------------------
def main():
    root = project_root_from_cwd()
    default_out = root / "Data" / "synthetic_dataset"

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=str, default=str(default_out),
                        help="Where to write dataset. Default: Data/synthetic_dataset")
    parser.add_argument("--n-patients", type=int, default=25)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    generate_dataset(
        output_root=Path(args.output_root),
        n_patients=args.n_patients,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
