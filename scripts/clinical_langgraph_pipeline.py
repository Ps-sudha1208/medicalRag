# """
# LangGraph Multi-Agent Clinical Risk Pipeline (FREE, Local) + LangChain Tools + LangSmith Tracing

# Implements 6-agent architecture:
# 1) Parser Agent
# 2) Validator Agent
# 3) Retrieval Agent (FAISS + MiniLM)
# 4) Temporal Analyzer Agent
# 5) Risk Assessor Agent (Rules + Optional RF + Optional Ollama LLM)
# 6) Response Generator Agent (Templates + Optional Ollama)

# Adds:
# - Pydantic state schema (Pydantic v2, no deprecated Config)
# - LangChain Tools (@tool) for extraction/retrieval/temporal/risk/response
# - LangGraph orchestration + 4-tier routing (LOW/MEDIUM/HIGH/CRITICAL)
# - LangSmith tracing (auto if LANGCHAIN_TRACING_V2 + LANGCHAIN_API_KEY env vars are set)
# - Modes:
#     - train_ml
#     - run_pdf
#     - run_folder  (batch folder -> results.jsonl + results.csv + summary + alerts)

# Examples:

# # (Optional) Train ML risk model from synthetic jsons
# python scripts/clinical_langgraph_pipeline.py --mode train_ml --dataset-root Data/synthetic_dataset

# # Run pipeline on a single PDF
# python scripts/clinical_langgraph_pipeline.py --mode run_pdf \
#   --pdf Data/synthetic_dataset/pdfs/PT0002_POD17_LOW_20260206.pdf \
#   --dataset-root Data/synthetic_dataset \
#   --faiss-dir Data/rag_artifacts/vectordb_faiss_minilm_384 \
#   --top-k 5 --debug

# # Run pipeline on a folder of PDFs + export results
# python scripts/clinical_langgraph_pipeline.py --mode run_folder \
#   --pdf-dir Data/synthetic_dataset/pdfs \
#   --out-dir results \
#   --dataset-root Data/synthetic_dataset \
#   --faiss-dir Data/rag_artifacts/vectordb_faiss_minilm_384 \
#   --top-k 5

# # Use Ollama (optional; requires ollama installed)
# #   ollama serve
# #   ollama pull mistral
# python scripts/clinical_langgraph_pipeline.py --mode run_pdf --pdf ... --use-ollama

# # LangSmith (optional):
# #   export LANGCHAIN_TRACING_V2=true
# #   export LANGCHAIN_API_KEY="ls__..."
# #   export LANGCHAIN_PROJECT="medicalRAG-clinical-risk"
# """

# from __future__ import annotations

# import argparse
# import csv
# import json
# import os
# import re
# from pathlib import Path
# from typing import Any, Dict, List, Optional, Tuple

# import numpy as np
# import requests
# import joblib
# from sklearn.ensemble import RandomForestClassifier

# from pydantic import BaseModel, Field, ConfigDict

# # -------------------------
# # LangGraph
# # -------------------------
# from langgraph.graph import StateGraph, END

# # -------------------------
# # LangChain tools + tracing config
# # -------------------------
# from langchain_core.tools import tool
# from langchain_core.runnables import RunnableConfig

# # -------------------------
# # LangChain PDF + FAISS + Embeddings
# # -------------------------
# try:
#     from langchain_community.document_loaders import PyPDFLoader
# except Exception:
#     from langchain.document_loaders import PyPDFLoader

# # Prefer langchain_huggingface if installed (removes deprecation warning)
# try:
#     from langchain_huggingface import HuggingFaceEmbeddings
# except Exception:
#     try:
#         from langchain_community.embeddings import HuggingFaceEmbeddings
#     except Exception:
#         from langchain.embeddings import HuggingFaceEmbeddings

# try:
#     from langchain_community.vectorstores import FAISS
# except Exception:
#     from langchain.vectorstores import FAISS


# # =========================================================
# # Constants
# # =========================================================
# MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# VALIDATION_RULES = {
#     "temperature": (35.0, 42.0),
#     "heart_rate": (40, 200),
#     "bp_systolic": (60, 220),
#     "bp_diastolic": (30, 140),
#     "respiratory_rate": (8, 40),
#     "spo2": (70, 100),
#     "wbc": (1.0, 50.0),
#     "hemoglobin": (4.0, 18.0),
#     "pain_score": (0, 10),
#     "pod": (0, 60),
# }

# RISK_TIERS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

# DEFAULT_DATASET_ROOT = Path("Data/synthetic_dataset")
# DEFAULT_FAISS_DIR = Path("Data/rag_artifacts/vectordb_faiss_minilm_384")
# DEFAULT_MODEL_DIR = Path("Data/rag_artifacts/models")
# DEFAULT_RF_PATH = DEFAULT_MODEL_DIR / "risk_rf.joblib"


# # =========================================================
# # Utilities
# # =========================================================
# def safe_float(x: Any) -> Optional[float]:
#     try:
#         return float(x)
#     except Exception:
#         return None


# def safe_int(x: Any) -> Optional[int]:
#     try:
#         return int(x)
#     except Exception:
#         return None


# def read_pdf_text(pdf_path: Path) -> str:
#     loader = PyPDFLoader(str(pdf_path))
#     docs = loader.load()
#     return "\n".join([d.page_content for d in docs])


# def parse_filename_for_pid_pod(name: str) -> Tuple[Optional[str], Optional[int]]:
#     # Example: PT0002_POD17_LOW_20260206.pdf
#     pid = None
#     pod = None
#     m1 = re.search(r"(PT\d{4})", name)
#     m2 = re.search(r"_POD(\d+)", name)
#     if m1:
#         pid = m1.group(1)
#     if m2:
#         pod = int(m2.group(1))
#     return pid, pod


# def ensure_dir(p: Path) -> None:
#     p.mkdir(parents=True, exist_ok=True)


# def summarize_counts(items: List[Dict[str, Any]]) -> Dict[str, int]:
#     counts = {k: 0 for k in RISK_TIERS}
#     for it in items:
#         lvl = it.get("risk_level")
#         if lvl in counts:
#             counts[lvl] += 1
#     return counts


# # =========================================================
# # Ollama helper (optional)
# # =========================================================
# def ollama_generate(prompt: str, model: str = "mistral", timeout: int = 60) -> Optional[str]:
#     """
#     Uses Ollama HTTP API if available (free, local).
#     Requires: ollama serve, and model pulled.
#     """
#     try:
#         resp = requests.post(
#             "http://localhost:11434/api/generate",
#             json={"model": model, "prompt": prompt, "stream": False},
#             timeout=timeout,
#         )
#         if resp.status_code != 200:
#             return None
#         data = resp.json()
#         return data.get("response")
#     except Exception:
#         return None


# # =========================================================
# # Pydantic State Schema (LangGraph-friendly)
# # =========================================================
# class PipelineState(BaseModel):
#     model_config = ConfigDict(arbitrary_types_allowed=True)

#     # Inputs/config
#     raw_text: str = ""
#     dataset_root: str = str(DEFAULT_DATASET_ROOT)
#     faiss_dir: str = str(DEFAULT_FAISS_DIR)
#     top_k: int = 5
#     use_ollama: bool = False
#     rf_path: str = str(DEFAULT_RF_PATH)
#     debug: bool = False

#     # optional metadata for tracing/outputs
#     pdf_name: str = ""
#     run_mode: str = ""  # run_pdf or run_folder

#     # Agent outputs
#     extracted_features: Dict[str, Any] = Field(default_factory=dict)
#     validated_features: Dict[str, Any] = Field(default_factory=dict)
#     validation: Dict[str, Any] = Field(default_factory=dict)

#     retrieved_context: Dict[str, Any] = Field(default_factory=dict)
#     temporal_analysis: Dict[str, Any] = Field(default_factory=dict)
#     risk_assessment: Dict[str, Any] = Field(default_factory=dict)
#     final_response: str = ""

#     # Tracing/errors
#     trace: List[str] = Field(default_factory=list)
#     errors: List[str] = Field(default_factory=list)


# def _trace(state: PipelineState, msg: str) -> None:
#     state.trace.append(msg)


# # =========================================================
# # LangChain Tools (@tool)
# # =========================================================

# @tool("extract_structured_features")
# def extract_structured_features_tool(raw_text: str) -> Dict[str, Any]:
#     """Extract structured clinical features from raw post-op text using regex heuristics."""
#     t = re.sub(r"[ \t]+", " ", raw_text or "")
#     t = re.sub(r"\n+", "\n", t)

#     def grab(pattern: str) -> Optional[str]:
#         m = re.search(pattern, t, flags=re.IGNORECASE | re.DOTALL)
#         return m.group(1).strip() if m else None

#     features: Dict[str, Any] = {}
#     features["patient_id"] = grab(r"Patient ID:\s*([A-Z]{2}\d{4})")
#     pod_str = grab(r"Post-Op Day\s*(\d+)")
#     features["pod"] = safe_int(pod_str) if pod_str else None
#     features["procedure"] = grab(r"Procedure:\s*([^\n]+)")
#     features["surgeon"] = grab(r"Surgeon:\s*([^\n]+)")
#     features["timestamp"] = grab(r"Timestamp:\s*([^\n]+)")

#     temp = grab(r"Temperature\s*([0-9]+(?:\.[0-9]+)?)")
#     hr = grab(r"Heart Rate\s*(\d+)")
#     bp = re.search(r"Blood Pressure\s*(\d{2,3})/(\d{2,3})", t, flags=re.IGNORECASE)
#     rr = grab(r"Respiratory Rate\s*(\d+)")
#     spo2 = grab(r"SpO2\s*(\d+)")
#     pain = grab(r"Pain Score\s*(\d+)")

#     features["temperature"] = safe_float(temp) if temp else None
#     features["heart_rate"] = safe_int(hr) if hr else None
#     if bp:
#         features["bp_systolic"] = safe_int(bp.group(1))
#         features["bp_diastolic"] = safe_int(bp.group(2))
#     else:
#         features["bp_systolic"] = None
#         features["bp_diastolic"] = None
#     features["respiratory_rate"] = safe_int(rr) if rr else None
#     features["spo2"] = safe_int(spo2) if spo2 else None
#     features["pain_score"] = safe_int(pain) if pain else None

#     wbc = grab(r"\bWBC\s*([0-9]+(?:\.[0-9]+)?)")
#     hgb = grab(r"Hemoglobin\s*([0-9]+(?:\.[0-9]+)?)")
#     features["wbc"] = safe_float(wbc) if wbc else None
#     features["hemoglobin"] = safe_float(hgb) if hgb else None

#     # (Optional future: platelets/creatinine; keep safe if absent)
#     platelets = grab(r"Platelets\s*([0-9]+(?:\.[0-9]+)?)")
#     creatinine = grab(r"Creatinine\s*([0-9]+(?:\.[0-9]+)?)")
#     features["platelets"] = safe_float(platelets) if platelets else None
#     features["creatinine"] = safe_float(creatinine) if creatinine else None

#     notes = grab(r"Clinical Notes\s*(.+?)Automated Risk")
#     if not notes:
#         notes = grab(r"Clinical Notes\s*(.+)$")
#     features["clinical_notes"] = notes

#     red_flags = re.findall(r"RED FLAG:\s*([A-Za-z \-]+)", t, flags=re.IGNORECASE)
#     features["red_flags"] = [rf.strip() for rf in red_flags] if red_flags else []

#     return features


# @tool("validate_features")
# def validate_features_tool(features: Dict[str, Any]) -> Dict[str, Any]:
#     """Validate physiological plausibility + required fields; returns validation + validated_features bundle."""
#     required = ["patient_id", "pod", "procedure", "temperature", "heart_rate", "spo2", "wbc", "pain_score"]
#     errors: List[str] = []
#     warnings: List[str] = []

#     for k in required:
#         if features.get(k) is None:
#             warnings.append(f"missing_required:{k}")

#     for k, (lo, hi) in VALIDATION_RULES.items():
#         v = features.get(k)
#         if v is None:
#             continue
#         try:
#             if not (lo <= float(v) <= hi):
#                 errors.append(f"out_of_range:{k}={v} not in [{lo},{hi}]")
#         except Exception:
#             errors.append(f"invalid_type:{k}={v}")

#     status = "ok"
#     if errors:
#         status = "error"
#     elif warnings:
#         status = "warn"

#     validated = dict(features)
#     validated["validator_status"] = status
#     validated["validator_errors"] = errors
#     validated["validator_warnings"] = warnings

#     return {
#         "validation": {"status": status, "errors": errors, "warnings": warnings},
#         "validated_features": validated,
#     }


# def craft_retrieval_query(vf: Dict[str, Any]) -> str:
#     parts = []
#     if vf.get("procedure"):
#         parts.append(f"Procedure: {vf['procedure']}")
#     if vf.get("pod") is not None:
#         parts.append(f"Post-Op Day {vf['pod']}")
#     for k in ["temperature", "heart_rate", "spo2", "wbc", "pain_score"]:
#         if vf.get(k) is not None:
#             parts.append(f"{k} {vf[k]}")
#     if vf.get("red_flags"):
#         parts.append("RED FLAG " + " ".join(vf["red_flags"]))
#     if vf.get("clinical_notes"):
#         parts.append(str(vf["clinical_notes"])[:200])
#     return " | ".join(parts) if parts else "post-operative progress note"


# def load_faiss_store(faiss_dir: Path):
#     embeddings = HuggingFaceEmbeddings(model_name=MINILM_MODEL)
#     return FAISS.load_local(str(faiss_dir), embeddings, allow_dangerous_deserialization=True)


# @tool("retrieve_context")
# def retrieve_context_tool(
#     validated_features: Dict[str, Any],
#     faiss_dir: str,
#     top_k: int = 5,
# ) -> Dict[str, Any]:
#     """Retrieve similar cases/context from FAISS using MiniLM embeddings."""
#     vf = validated_features or {}
#     faiss_path = Path(faiss_dir).resolve()
#     query = craft_retrieval_query(vf)

#     if not faiss_path.exists():
#         return {"query_used": query, "top_k": 0, "citations": [], "contexts": [], "warning": f"faiss_dir_missing:{faiss_path}"}

#     store = load_faiss_store(faiss_path)
#     retriever = store.as_retriever(search_kwargs={"k": int(top_k)})
#     docs = retriever.invoke(query) if hasattr(retriever, "invoke") else retriever.get_relevant_documents(query)

#     contexts = []
#     citations = []
#     for d in docs:
#         meta = d.metadata or {}
#         src = meta.get("source_file") or meta.get("source") or "unknown"
#         page = meta.get("page")
#         contexts.append({"source": src, "page": page, "text": d.page_content})
#         citations.append(f"{src}" + (f":p{page}" if page is not None else ""))

#     return {
#         "query_used": query,
#         "top_k": int(top_k),
#         "citations": citations,
#         "contexts": contexts,
#     }


# def load_patient_history_from_jsons(dataset_root: Path, patient_id: str, current_pod: int, lookback: int = 3):
#     json_dir = dataset_root / "jsons"
#     if not json_dir.exists():
#         return []

#     history = []
#     for jp in sorted(json_dir.glob("*.json")):
#         pid, pod = parse_filename_for_pid_pod(jp.name)
#         if pid != patient_id or pod is None:
#             continue
#         if pod < current_pod and pod >= max(0, current_pod - lookback):
#             try:
#                 data = json.loads(jp.read_text(encoding="utf-8"))
#                 history.append(data.get("daily_observation", {}))
#             except Exception:
#                 pass
#     history.sort(key=lambda x: x.get("pod", 0))
#     return history


# def trend(values: List[float]) -> str:
#     if len(values) < 2:
#         return "insufficient_history"
#     if values[-1] > values[0] + 0.2:
#         return "worsening"
#     if values[-1] < values[0] - 0.2:
#         return "improving"
#     return "stable"


# @tool("temporal_analysis")
# def temporal_analysis_tool(validated_features: Dict[str, Any], dataset_root: str) -> Dict[str, Any]:
#     """Analyze trends over last N PODs using JSON sidecars (synthetic history)."""
#     vf = validated_features or {}
#     pid = vf.get("patient_id")
#     pod = vf.get("pod")

#     if not pid or pod is None:
#         return {"status": "warn", "reason": "missing patient_id/pod"}

#     root = Path(dataset_root).resolve()
#     hist = load_patient_history_from_jsons(root, str(pid), int(pod), lookback=3)

#     def extract_series(key: str) -> List[float]:
#         vals: List[float] = []
#         for h in hist:
#             v = h.get(key)
#             if v is not None:
#                 try:
#                     vals.append(float(v))
#                 except Exception:
#                     pass
#         cv = vf.get(key)
#         if cv is not None:
#             try:
#                 vals.append(float(cv))
#             except Exception:
#                 pass
#         return vals

#     return {
#         "status": "ok" if hist else "warn",
#         "history_points_used": len(hist),
#         "temperature_trend": trend(extract_series("temperature")),
#         "heart_rate_trend": trend(extract_series("heart_rate")),
#         "wbc_trend": trend(extract_series("wbc")),
#         "spo2_trend": trend(extract_series("spo2")),
#         "notes": "trend uses last 3 PODs when available; otherwise single-point",
#     }


# def rule_risk_score(vf: Dict[str, Any]) -> Tuple[int, str, List[str]]:
#     score = 0
#     factors: List[str] = []

#     temp = vf.get("temperature") or 0
#     hr = vf.get("heart_rate") or 0
#     bps = vf.get("bp_systolic") or 120
#     spo2 = vf.get("spo2") or 100
#     wbc = vf.get("wbc") or 7
#     pain = vf.get("pain_score") or 0
#     notes = (vf.get("clinical_notes") or "").lower()
#     red_flags = [str(rf).lower() for rf in (vf.get("red_flags") or [])]

#     if temp > 39.0:
#         score += 30
#         factors.append("temp>39")
#     elif temp > 38.0:
#         score += 15
#         factors.append("temp>38")

#     if hr > 120:
#         score += 25
#         factors.append("hr>120")
#     elif hr > 100:
#         score += 15
#         factors.append("hr>100")

#     if bps < 90:
#         score += 25
#         factors.append("sbp<90")

#     if spo2 < 90:
#         score += 30
#         factors.append("spo2<90")
#     elif spo2 < 95:
#         score += 15
#         factors.append("spo2<95")

#     if wbc > 15.0:
#         score += 20
#         factors.append("wbc>15")
#     elif wbc > 11.0:
#         score += 10
#         factors.append("wbc>11")

#     if pain > 7:
#         score += 15
#         factors.append("pain>7")
#     elif pain > 5:
#         score += 5
#         factors.append("pain>5")

#     for flag in ["sepsis", "wound dehiscence", "pulmonary embolism", "hemorrhage", "shock"]:
#         if flag in notes or flag in " ".join(red_flags):
#             score += 40
#             factors.append(f"red_flag:{flag}")

#     if score >= 76:
#         level = "CRITICAL"
#     elif score >= 51:
#         level = "HIGH"
#     elif score >= 26:
#         level = "MEDIUM"
#     else:
#         level = "LOW"
#     return int(score), level, factors


# def ml_features_vector(vf: Dict[str, Any], temporal: Dict[str, Any]) -> np.ndarray:
#     def tcode(x: str) -> int:
#         return {"improving": -1, "stable": 0, "worsening": 1}.get(x, 0)

#     vec = [
#         float(vf.get("temperature") or 0),
#         float(vf.get("heart_rate") or 0),
#         float(vf.get("bp_systolic") or 0),
#         float(vf.get("spo2") or 0),
#         float(vf.get("wbc") or 0),
#         float(vf.get("hemoglobin") or 0),
#         float(vf.get("pain_score") or 0),
#         float(vf.get("pod") or 0),
#         tcode((temporal or {}).get("temperature_trend", "stable")),
#         tcode((temporal or {}).get("wbc_trend", "stable")),
#         len(vf.get("red_flags") or []),
#     ]
#     return np.array(vec, dtype=np.float32).reshape(1, -1)


# @tool("risk_assessment")
# def risk_assessment_tool(
#     validated_features: Dict[str, Any],
#     temporal_analysis: Dict[str, Any],
#     retrieved_context: Dict[str, Any],
#     rf_path: str,
#     use_ollama: bool = False,
# ) -> Dict[str, Any]:
#     """Compute risk using rules + optional RF model + optional Ollama reasoning."""
#     vf = validated_features or {}
#     temporal = temporal_analysis or {}
#     retrieved = retrieved_context or {}

#     # Layer 1: Rules
#     rule_score, rule_level, rule_factors = rule_risk_score(vf)

#     # Layer 2: ML (optional)
#     ml_pred = None
#     ml_proba = None
#     rf_file = Path(rf_path).resolve()
#     if rf_file.exists():
#         try:
#             rf = joblib.load(rf_file)
#             X = ml_features_vector(vf, temporal)
#             ml_pred = rf.predict(X)[0]
#             if hasattr(rf, "predict_proba"):
#                 probs = rf.predict_proba(X)[0]
#                 ml_proba = {str(rf.classes_[i]): float(probs[i]) for i in range(len(rf.classes_))}
#         except Exception as e:
#             # ML fails -> continue with rules
#             ml_pred = None
#             ml_proba = None

#     # Layer 3: LLM (optional)
#     llm_reasoning = None
#     if use_ollama:
#         prompt = f"""
# You are a clinical risk assistant for post-op monitoring.
# Given:
# - validated_features: {json.dumps(vf, indent=2)}
# - temporal_analysis: {json.dumps(temporal, indent=2)}
# - retrieved_context_citations: {retrieved.get('citations', [])}

# Produce:
# 1) concise risk reasoning (2-4 bullet points)
# 2) likely complications to consider
# 3) any urgent alerts
# Return plain text.
# """
#         llm_reasoning = ollama_generate(prompt, model="mistral", timeout=60)

#     # Combine conservatively: escalate if ML higher than rules
#     final_level = rule_level
#     final_score = rule_score
#     factors = list(rule_factors)

#     if ml_pred in RISK_TIERS:
#         if RISK_TIERS.index(ml_pred) > RISK_TIERS.index(final_level):
#             final_level = ml_pred
#             factors.append(f"ml_escalation:{ml_pred}")

#     alerts: List[str] = []
#     if final_level in ["HIGH", "CRITICAL"]:
#         alerts.append("Urgent clinical review recommended")
#     if any("red_flag:" in f for f in factors):
#         alerts.append("RED FLAG detected in notes")

#     return {
#         "risk_level": final_level,
#         "risk_score": final_score,
#         "alerts": alerts,
#         "rule_score": rule_score,
#         "rule_level": rule_level,
#         "rule_factors": rule_factors,
#         "ml_pred": ml_pred,
#         "ml_proba": ml_proba,
#         "final_factors": factors,
#         "llm_reasoning": llm_reasoning,
#     }


# def template_response(level: str, vf: Dict[str, Any], temporal: Dict[str, Any], risk: Dict[str, Any], citations: List[str]) -> str:
#     alerts = risk.get("alerts", [])
#     pid = vf.get("patient_id", "UNKNOWN")
#     pod = vf.get("pod", "NA")
#     proc = vf.get("procedure", "NA")

#     lines: List[str] = []
#     lines.append(f"Clinical Assessment (Synthetic Demo) — Patient {pid} | POD {pod} | Procedure: {proc}")

#     lines.append("\nSummary:")
#     lines.append(f"- Risk Tier: {level} (risk_score={risk.get('risk_score')}, ml_pred={risk.get('ml_pred')})")
#     lines.append(
#         f"- Vitals: Temp={vf.get('temperature')} HR={vf.get('heart_rate')} SBP={vf.get('bp_systolic')} SpO2={vf.get('spo2')}"
#     )
#     lines.append(f"- Labs: WBC={vf.get('wbc')} Hgb={vf.get('hemoglobin')} Pain={vf.get('pain_score')}")

#     lines.append("\nTrends:")
#     lines.append(f"- Temp trend: {temporal.get('temperature_trend')}")
#     lines.append(f"- WBC trend:  {temporal.get('wbc_trend')}")
#     lines.append(f"- HR trend:   {temporal.get('heart_rate_trend')}")
#     lines.append(f"- SpO2 trend: {temporal.get('spo2_trend')}")

#     lines.append("\nFactors:")
#     ff = risk.get("final_factors") or risk.get("rule_factors") or []
#     if not ff:
#         lines.append("- (none)")
#     else:
#         for f in ff[:8]:
#             lines.append(f"- {f}")

#     if alerts:
#         lines.append("\nALERTS:")
#         for a in alerts:
#             lines.append(f"- {a}")

#     lines.append("\nRecommendations:")
#     if level == "LOW":
#         lines.append("- Continue routine post-op monitoring and reassess as scheduled.")
#     elif level == "MEDIUM":
#         lines.append("- Repeat vitals/labs, monitor for infection/atelectasis, reassess within 4–6 hours.")
#     elif level == "HIGH":
#         lines.append("- Urgent clinician review. Consider cultures/imaging; screen for sepsis if suspected.")
#     else:
#         lines.append("- Immediate escalation. Activate emergency pathway (sepsis/PE/hemorrhage consideration).")

#     if citations:
#         lines.append("\nRetrieved supporting notes (citations):")
#         for c in citations[:8]:
#             lines.append(f"- {c}")

#     lines.append("\nDisclaimer: Synthetic data for testing only. Not medical advice.")
#     return "\n".join(lines)


# @tool("generate_response")
# def generate_response_tool(
#     validated_features: Dict[str, Any],
#     temporal_analysis: Dict[str, Any],
#     risk_assessment: Dict[str, Any],
#     retrieved_context: Dict[str, Any],
#     use_ollama: bool = False,
#     response_style: str = "default",
# ) -> str:
#     """Generate clinician-friendly response using templates + optional Ollama polishing."""
#     vf = validated_features or {}
#     temporal = temporal_analysis or {}
#     risk = risk_assessment or {}
#     citations = (retrieved_context or {}).get("citations") or []
#     level = risk.get("risk_level", "MEDIUM")

#     base = template_response(level, vf, temporal, risk, citations)

#     if use_ollama:
#         prompt = f"""
# Rewrite the following assessment to be concise and clinician-friendly.
# Do NOT add new facts. Keep the same risk tier and recommendations.
# Text:
# {base}
# """
#         polished = ollama_generate(prompt, model="mistral", timeout=60)
#         if polished and len(polished.strip()) > 50:
#             return polished.strip()

#     return base


# # =========================================================
# # LangGraph Nodes (agents) calling tools
# # =========================================================
# def agent_parser(state: PipelineState, config: Optional[RunnableConfig] = None) -> PipelineState:
#     _trace(state, "agent_parser:start")
#     try:
#         feats = extract_structured_features_tool.invoke({"raw_text": state.raw_text}, config=config)
#         state.extracted_features = feats
#     except Exception as e:
#         state.errors.append(f"agent_parser_failed:{e}")
#     _trace(state, "agent_parser:done")
#     return state


# def agent_validator(state: PipelineState, config: Optional[RunnableConfig] = None) -> PipelineState:
#     _trace(state, "agent_validator:start")
#     try:
#         out = validate_features_tool.invoke({"features": state.extracted_features}, config=config)
#         state.validation = out.get("validation", {})
#         state.validated_features = out.get("validated_features", {})
#     except Exception as e:
#         state.errors.append(f"agent_validator_failed:{e}")
#     _trace(state, "agent_validator:done")
#     return state


# def agent_retrieval(state: PipelineState, config: Optional[RunnableConfig] = None) -> PipelineState:
#     _trace(state, "agent_retrieval:start")
#     try:
#         rc = retrieve_context_tool.invoke(
#             {"validated_features": state.validated_features, "faiss_dir": state.faiss_dir, "top_k": state.top_k},
#             config=config,
#         )
#         state.retrieved_context = rc
#         if rc.get("warning"):
#             state.errors.append(f"agent_retrieval_warning:{rc.get('warning')}")
#     except Exception as e:
#         # capture, continue
#         state.errors.append(f"agent_retrieval_failed:{e}")
#         state.retrieved_context = {"query_used": craft_retrieval_query(state.validated_features), "top_k": 0, "citations": [], "contexts": []}
#     _trace(state, "agent_retrieval:done")
#     return state


# def agent_temporal(state: PipelineState, config: Optional[RunnableConfig] = None) -> PipelineState:
#     _trace(state, "agent_temporal:start")
#     try:
#         ta = temporal_analysis_tool.invoke({"validated_features": state.validated_features, "dataset_root": state.dataset_root}, config=config)
#         state.temporal_analysis = ta
#     except Exception as e:
#         state.errors.append(f"agent_temporal_failed:{e}")
#         state.temporal_analysis = {"status": "warn", "reason": str(e)}
#     _trace(state, "agent_temporal:done")
#     return state


# def agent_risk_assessor(state: PipelineState, config: Optional[RunnableConfig] = None) -> PipelineState:
#     _trace(state, "agent_risk_assessor:start")
#     try:
#         ra = risk_assessment_tool.invoke(
#             {
#                 "validated_features": state.validated_features,
#                 "temporal_analysis": state.temporal_analysis,
#                 "retrieved_context": state.retrieved_context,
#                 "rf_path": state.rf_path,
#                 "use_ollama": state.use_ollama,
#             },
#             config=config,
#         )
#         state.risk_assessment = ra
#     except Exception as e:
#         state.errors.append(f"agent_risk_assessor_failed:{e}")
#         # conservative fallback
#         state.risk_assessment = {"risk_level": "MEDIUM", "risk_score": 0, "alerts": ["Risk engine failed; defaulting to MEDIUM"], "final_factors": []}
#     _trace(state, "agent_risk_assessor:done")
#     return state


# # --- 4-tier routing response nodes ---
# def agent_respond_low(state: PipelineState, config: Optional[RunnableConfig] = None) -> PipelineState:
#     _trace(state, "agent_respond_low:start")
#     state.final_response = generate_response_tool.invoke(
#         {
#             "validated_features": state.validated_features,
#             "temporal_analysis": state.temporal_analysis,
#             "risk_assessment": state.risk_assessment,
#             "retrieved_context": state.retrieved_context,
#             "use_ollama": state.use_ollama,
#             "response_style": "low",
#         },
#         config=config,
#     )
#     _trace(state, "agent_respond_low:done")
#     return state


# def agent_respond_medium(state: PipelineState, config: Optional[RunnableConfig] = None) -> PipelineState:
#     _trace(state, "agent_respond_medium:start")
#     state.final_response = generate_response_tool.invoke(
#         {
#             "validated_features": state.validated_features,
#             "temporal_analysis": state.temporal_analysis,
#             "risk_assessment": state.risk_assessment,
#             "retrieved_context": state.retrieved_context,
#             "use_ollama": state.use_ollama,
#             "response_style": "medium",
#         },
#         config=config,
#     )
#     _trace(state, "agent_respond_medium:done")
#     return state


# def agent_respond_high(state: PipelineState, config: Optional[RunnableConfig] = None) -> PipelineState:
#     _trace(state, "agent_respond_high:start")
#     state.final_response = generate_response_tool.invoke(
#         {
#             "validated_features": state.validated_features,
#             "temporal_analysis": state.temporal_analysis,
#             "risk_assessment": state.risk_assessment,
#             "retrieved_context": state.retrieved_context,
#             "use_ollama": state.use_ollama,
#             "response_style": "high",
#         },
#         config=config,
#     )
#     _trace(state, "agent_respond_high:done")
#     return state


# def agent_respond_critical(state: PipelineState, config: Optional[RunnableConfig] = None) -> PipelineState:
#     _trace(state, "agent_respond_critical:start")
#     state.final_response = generate_response_tool.invoke(
#         {
#             "validated_features": state.validated_features,
#             "temporal_analysis": state.temporal_analysis,
#             "risk_assessment": state.risk_assessment,
#             "retrieved_context": state.retrieved_context,
#             "use_ollama": state.use_ollama,
#             "response_style": "critical",
#         },
#         config=config,
#     )
#     _trace(state, "agent_respond_critical:done")
#     return state


# def route_after_risk(state: PipelineState) -> str:
#     level = (state.risk_assessment or {}).get("risk_level", "MEDIUM")
#     if level == "LOW":
#         return "respond_low"
#     if level == "MEDIUM":
#         return "respond_medium"
#     if level == "HIGH":
#         return "respond_high"
#     return "respond_critical"


# def build_graph():
#     g = StateGraph(PipelineState)

#     g.add_node("parse", agent_parser)
#     g.add_node("validate", agent_validator)
#     g.add_node("retrieve", agent_retrieval)
#     g.add_node("temporal", agent_temporal)
#     g.add_node("assess_risk", agent_risk_assessor)

#     g.add_node("respond_low", agent_respond_low)
#     g.add_node("respond_medium", agent_respond_medium)
#     g.add_node("respond_high", agent_respond_high)
#     g.add_node("respond_critical", agent_respond_critical)

#     g.set_entry_point("parse")

#     g.add_edge("parse", "validate")
#     g.add_edge("validate", "retrieve")
#     g.add_edge("retrieve", "temporal")
#     g.add_edge("temporal", "assess_risk")

#     g.add_conditional_edges(
#         "assess_risk",
#         route_after_risk,
#         {
#             "respond_low": "respond_low",
#             "respond_medium": "respond_medium",
#             "respond_high": "respond_high",
#             "respond_critical": "respond_critical",
#         },
#     )

#     g.add_edge("respond_low", END)
#     g.add_edge("respond_medium", END)
#     g.add_edge("respond_high", END)
#     g.add_edge("respond_critical", END)

#     return g.compile()


# # =========================================================
# # ML training
# # =========================================================
# def train_random_forest(dataset_root: Path, out_path: Path):
#     json_dir = dataset_root / "jsons"
#     if not json_dir.exists():
#         raise FileNotFoundError(f"Missing json dir: {json_dir}")

#     X_list = []
#     y_list = []

#     for jp in json_dir.glob("*.json"):
#         try:
#             data = json.loads(jp.read_text(encoding="utf-8"))
#             dob = data.get("daily_observation", {})
#             gt = data.get("ground_truth", {})

#             vf = {
#                 "temperature": dob.get("temperature"),
#                 "heart_rate": dob.get("heart_rate"),
#                 "bp_systolic": safe_int(str(dob.get("blood_pressure", "120/80")).split("/")[0]) if dob.get("blood_pressure") else 120,
#                 "spo2": dob.get("spo2"),
#                 "wbc": dob.get("wbc"),
#                 "hemoglobin": dob.get("hemoglobin"),
#                 "pain_score": dob.get("pain_score"),
#                 "pod": dob.get("pod"),
#                 "red_flags": re.findall(r"RED FLAG:\s*([A-Za-z \-]+)", (dob.get("clinical_notes") or ""), flags=re.IGNORECASE),
#             }

#             temporal = {"temperature_trend": "stable", "wbc_trend": "stable"}
#             X = ml_features_vector(vf, temporal).reshape(-1)
#             y = gt.get("risk_level_rule") or dob.get("rule_risk_level")

#             if y not in RISK_TIERS:
#                 continue

#             X_list.append(X)
#             y_list.append(y)
#         except Exception:
#             continue

#     if len(X_list) < 20:
#         raise RuntimeError(f"Not enough training samples found in {json_dir} (found {len(X_list)})")

#     X_arr = np.vstack(X_list)
#     y_arr = np.array(y_list)

#     rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
#     rf.fit(X_arr, y_arr)

#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     joblib.dump(rf, out_path)
#     print(f"Trained RandomForest on {len(X_arr)} samples")
#     print(f"Saved model to: {out_path}")


# # =========================================================
# # Run helpers
# # =========================================================
# def _langsmith_enabled() -> bool:
#     # LangSmith is enabled when tracing env is on and api key exists.
#     v = os.getenv("LANGCHAIN_TRACING_V2", "").lower() in ("1", "true", "yes", "on")
#     k = bool(os.getenv("LANGCHAIN_API_KEY"))
#     return v and k


# def make_run_config(run_name: str, tags: List[str], metadata: Dict[str, Any]) -> RunnableConfig:
#     # LangSmith will pick this up when env vars are set.
#     return {
#         "run_name": run_name,
#         "tags": tags,
#         "metadata": metadata,
#     }


# def run_one_pdf(graph, pdf_path: Path, args) -> PipelineState:
#     raw_text = read_pdf_text(pdf_path)

#     init_state = PipelineState(
#         raw_text=raw_text,
#         dataset_root=str(Path(args.dataset_root).resolve()),
#         faiss_dir=str(Path(args.faiss_dir).resolve()),
#         top_k=int(args.top_k),
#         use_ollama=bool(args.use_ollama),
#         rf_path=str(DEFAULT_RF_PATH),
#         debug=bool(args.debug),
#         pdf_name=pdf_path.name,
#         run_mode=args.mode,
#     )

#     # best-effort run_name for tracing
#     pid, pod = parse_filename_for_pid_pod(pdf_path.name)
#     run_name = f"{pid or 'UNKNOWN'}_POD{pod if pod is not None else 'NA'}__{pdf_path.stem}"
#     tags = [args.mode]
#     metadata = {
#         "pdf": pdf_path.name,
#         "patient_id_guess": pid,
#         "pod_guess": pod,
#         "faiss_dir": str(Path(args.faiss_dir).resolve()),
#         "dataset_root": str(Path(args.dataset_root).resolve()),
#         "top_k": int(args.top_k),
#         "use_ollama": bool(args.use_ollama),
#     }
#     config = make_run_config(run_name, tags, metadata) if _langsmith_enabled() else None

#     out = graph.invoke(init_state, config=config)

#     # LangGraph may return dict OR PipelineState depending on version
#     if isinstance(out, PipelineState):
#         return out
#     if isinstance(out, dict):
#         return PipelineState(**out)
#     return PipelineState()


# def run_folder(args) -> None:
#     pdf_dir = Path(args.pdf_dir).resolve()
#     if not pdf_dir.exists():
#         raise FileNotFoundError(f"--pdf-dir not found: {pdf_dir}")

#     out_dir = Path(args.out_dir).resolve()
#     ensure_dir(out_dir)
#     jsonl_path = out_dir / "results.jsonl"
#     csv_path = out_dir / "results.csv"

#     graph = build_graph()

#     pdfs = sorted(list(pdf_dir.glob("*.pdf")))
#     if not pdfs:
#         raise RuntimeError(f"No PDFs found in {pdf_dir}")

#     results: List[Dict[str, Any]] = []
#     alerts_all: List[str] = []

#     with jsonl_path.open("w", encoding="utf-8") as jf:
#         for pdf in pdfs:
#             st = run_one_pdf(graph, pdf, args)

#             vf = st.validated_features or {}
#             risk = st.risk_assessment or {}
#             rc = st.retrieved_context or {}

#             row = {
#                 "pdf": pdf.name,
#                 "patient_id": vf.get("patient_id"),
#                 "pod": vf.get("pod"),
#                 "procedure": vf.get("procedure"),
#                 "risk_level": risk.get("risk_level"),
#                 "risk_score": risk.get("risk_score"),
#                 "alerts": risk.get("alerts") or [],
#                 "citations": rc.get("citations") or [],
#                 "retrieval_query_used": rc.get("query_used"),
#                 "errors": st.errors,
#             }
#             results.append(row)
#             jf.write(json.dumps(row) + "\n")

#             for a in (row["alerts"] or []):
#                 alerts_all.append(f"{pdf.name} | {row.get('patient_id')} | {row.get('risk_level')} | {a}")

#     # Write CSV
#     with csv_path.open("w", newline="", encoding="utf-8") as cf:
#         writer = csv.DictWriter(
#             cf,
#             fieldnames=[
#                 "pdf",
#                 "patient_id",
#                 "pod",
#                 "procedure",
#                 "risk_level",
#                 "risk_score",
#                 "alerts",
#                 "citations",
#                 "retrieval_query_used",
#                 "errors",
#             ],
#         )
#         writer.writeheader()
#         for r in results:
#             r2 = dict(r)
#             r2["alerts"] = "; ".join(r2.get("alerts") or [])
#             r2["citations"] = "; ".join(r2.get("citations") or [])
#             r2["errors"] = "; ".join(r2.get("errors") or [])
#             writer.writerow(r2)

#     counts = summarize_counts(results)

#     print("\n================ FOLDER SUMMARY ================\n")
#     print(f"Processed PDFs: {len(results)}")
#     print("Risk tier counts:")
#     for k in RISK_TIERS:
#         print(f"  {k:8s}: {counts.get(k, 0)}")

#     if alerts_all:
#         print("\nAlerts:")
#         for line in alerts_all[:50]:
#             print(" -", line)
#         if len(alerts_all) > 50:
#             print(f" ... ({len(alerts_all) - 50} more)")
#     else:
#         print("\nAlerts: (none)")

#     print(f"\nSaved JSONL: {jsonl_path}")
#     print(f"Saved CSV:   {csv_path}")
#     print("\n===============================================\n")


# # =========================================================
# # CLI
# # =========================================================
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--mode", type=str, required=True, choices=["train_ml", "run_pdf", "run_folder"])

#     parser.add_argument("--pdf", type=str, default=None)
#     parser.add_argument("--pdf-dir", type=str, default=None, help="Folder of PDFs (for mode=run_folder)")
#     parser.add_argument("--out-dir", type=str, default="results", help="Output folder for mode=run_folder")

#     parser.add_argument("--dataset-root", type=str, default=str(DEFAULT_DATASET_ROOT))

#     # Support both --faiss-dir and --faiss-index (alias)
#     parser.add_argument("--faiss-dir", type=str, default=str(DEFAULT_FAISS_DIR))
#     parser.add_argument("--faiss-index", type=str, default=None, help="Alias for --faiss-dir")

#     parser.add_argument("--top-k", type=int, default=5)
#     parser.add_argument("--use-ollama", action="store_true")
#     parser.add_argument("--debug", action="store_true")

#     args = parser.parse_args()

#     # normalize alias
#     if args.faiss_index:
#         args.faiss_dir = args.faiss_index

#     dataset_root = Path(args.dataset_root).resolve()

#     if args.mode == "train_ml":
#         train_random_forest(dataset_root, DEFAULT_RF_PATH)
#         return

#     if args.mode == "run_pdf":
#         if not args.pdf:
#             raise ValueError("--pdf is required for mode=run_pdf")
#         pdf_path = Path(args.pdf).resolve()
#         if not pdf_path.exists():
#             raise FileNotFoundError(f"PDF not found: {pdf_path}")

#         graph = build_graph()
#         st = run_one_pdf(graph, pdf_path, args)

#         if args.debug:
#             print("\n---- DEBUG TRACE ----")
#             print(st.trace)
#             print("\n---- DEBUG ERRORS ----")
#             print(st.errors)

#         print("\n================ FINAL OUTPUT ================\n")
#         print(st.final_response or "NO RESPONSE")
#         print("\n=============================================\n")

#         if args.debug:
#             print("validation:", json.dumps(st.validation, indent=2))
#             print("risk:", json.dumps(st.risk_assessment, indent=2))
#             print("retrieval_query_used:", (st.retrieved_context or {}).get("query_used"))
#             print("citations:", (st.retrieved_context or {}).get("citations"))

#         return

#     if args.mode == "run_folder":
#         if not args.pdf_dir:
#             raise ValueError("--pdf-dir is required for mode=run_folder")
#         run_folder(args)
#         return


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""
LangGraph Multi-Agent Clinical Risk Pipeline (FREE, Local)
+ LangChain Tools + LangSmith Tracing
+ NEW: Parser fallbacks (quality score → spaCy → Ollama extraction)
+ NEW: Retrieval expansion (similar_cases + guidelines + patient_history)
+ NEW: Temporal upgrades (expected POD ranges + rapid deterioration rule)

Implements 6-agent architecture:
1) Parser Agent
2) Validator Agent
3) Retrieval Agent (FAISS + MiniLM)  [expanded]
4) Temporal Analyzer Agent           [upgraded]
5) Risk Assessor Agent (Rules + Optional RF + Optional Ollama LLM)
6) Response Generator Agent (Templates + Optional Ollama)

LangSmith:
- auto enabled if LANGCHAIN_TRACING_V2 + LANGCHAIN_API_KEY env vars are set
- run_name/tags/metadata set at graph.invoke
- tool invocations also get enriched metadata (patient_id/pod/risk where possible)

CLI modes:
  - train_ml
  - run_pdf
  - run_folder

Examples:

# Train ML model (optional)
python scripts/clinical_langgraph_pipeline.py --mode train_ml --dataset-root Data/synthetic_dataset

# Run pipeline on one PDF
python scripts/clinical_langgraph_pipeline.py --mode run_pdf \
  --pdf Data/synthetic_dataset/pdfs/PT0002_POD17_LOW_20260206.pdf \
  --dataset-root Data/synthetic_dataset \
  --faiss-cases-dir Data/rag_artifacts/vectordb_faiss_minilm_384 \
  --faiss-guidelines-dir Data/rag_artifacts/vectordb_guidelines_minilm_384 \
  --top-k 5 --debug

# Run folder
python scripts/clinical_langgraph_pipeline.py --mode run_folder \
  --pdf-dir Data/synthetic_dataset/pdfs \
  --out-dir results \
  --dataset-root Data/synthetic_dataset \
  --faiss-cases-dir Data/rag_artifacts/vectordb_faiss_minilm_384 \
  --faiss-guidelines-dir Data/rag_artifacts/vectordb_guidelines_minilm_384 \
  --top-k 5

# Use Ollama (optional)
#   ollama serve
#   ollama pull mistral
python scripts/clinical_langgraph_pipeline.py --mode run_pdf --pdf ... --use-ollama

Notes:
- spaCy fallback is optional:
    pip install spacy
    python -m spacy download en_core_web_sm
- guideline index is optional; if missing, retrieval returns warning but pipeline continues.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import requests
from pydantic import BaseModel, ConfigDict, Field
from sklearn.ensemble import RandomForestClassifier

from langgraph.graph import END, StateGraph
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

# -------------------------
# LangChain PDF + FAISS + Embeddings
# -------------------------
try:
    from langchain_community.document_loaders import PyPDFLoader
except Exception:
    from langchain.document_loaders import PyPDFLoader

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except Exception:
        from langchain.embeddings import HuggingFaceEmbeddings

try:
    from langchain_community.vectorstores import FAISS
except Exception:
    from langchain.vectorstores import FAISS

# -------------------------
# Optional spaCy
# -------------------------
try:
    import spacy  # type: ignore
except Exception:
    spacy = None


# =========================================================
# Constants
# =========================================================
MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RISK_TIERS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

VALIDATION_RULES = {
    "temperature": (35.0, 42.0),
    "heart_rate": (40, 200),
    "bp_systolic": (60, 220),
    "bp_diastolic": (30, 140),
    "respiratory_rate": (8, 40),
    "spo2": (70, 100),
    "wbc": (1.0, 50.0),
    "hemoglobin": (4.0, 18.0),
    "pain_score": (0, 10),
    "pod": (0, 60),
}

DEFAULT_DATASET_ROOT = Path("Data/synthetic_dataset")
DEFAULT_FAISS_CASES_DIR = Path("Data/rag_artifacts/vectordb_faiss_minilm_384")
DEFAULT_FAISS_GUIDELINES_DIR = Path("Data/rag_artifacts/vectordb_guidelines_minilm_384")
DEFAULT_MODEL_DIR = Path("Data/rag_artifacts/models")
DEFAULT_RF_PATH = DEFAULT_MODEL_DIR / "risk_rf.joblib"

REQUIRED_FOR_QUALITY = ["patient_id", "pod", "procedure", "temperature", "heart_rate", "spo2", "wbc", "pain_score"]
QUALITY_THRESHOLD = 0.70  # <70% required fields -> fallback

# Heuristic expected ranges by POD bucket (demo-level rules you can refine)
# Buckets: early (0-3), mid (4-7), late (8+)
EXPECTED_RANGES_DEFAULT = {
    "early": {"temperature": (36.0, 38.5), "heart_rate": (60, 110), "wbc": (4.0, 14.0), "spo2": (92, 100)},
    "mid": {"temperature": (36.0, 38.0), "heart_rate": (55, 100), "wbc": (4.0, 12.0), "spo2": (94, 100)},
    "late": {"temperature": (36.0, 37.8), "heart_rate": (50, 95), "wbc": (4.0, 11.0), "spo2": (95, 100)},
}

# Optional per-procedure overrides (extend later)
EXPECTED_RANGES_BY_PROCEDURE = {
    "appendectomy": EXPECTED_RANGES_DEFAULT,
    "cholecystectomy": EXPECTED_RANGES_DEFAULT,
    "colectomy": {
        "early": {"temperature": (36.0, 38.8), "heart_rate": (60, 120), "wbc": (4.0, 16.0), "spo2": (92, 100)},
        "mid": {"temperature": (36.0, 38.2), "heart_rate": (55, 110), "wbc": (4.0, 13.0), "spo2": (93, 100)},
        "late": {"temperature": (36.0, 37.9), "heart_rate": (50, 100), "wbc": (4.0, 11.5), "spo2": (94, 100)},
    },
}


# =========================================================
# Utilities
# =========================================================
def safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def safe_int(x: Any) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


def read_pdf_text(pdf_path: Path) -> str:
    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()
    return "\n".join([d.page_content for d in docs])


def parse_filename_for_pid_pod(name: str) -> Tuple[Optional[str], Optional[int]]:
    pid = None
    pod = None
    m1 = re.search(r"(PT\d{4})", name)
    m2 = re.search(r"_POD(\d+)", name)
    if m1:
        pid = m1.group(1)
    if m2:
        pod = int(m2.group(1))
    return pid, pod


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def summarize_counts(items: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {k: 0 for k in RISK_TIERS}
    for it in items:
        lvl = it.get("risk_level")
        if lvl in counts:
            counts[lvl] += 1
    return counts


def quality_score(features: Dict[str, Any], required_keys: List[str]) -> float:
    if not required_keys:
        return 1.0
    present = sum(1 for k in required_keys if features.get(k) is not None)
    return present / float(len(required_keys))


def normalize_procedure(proc: Optional[str]) -> str:
    p = (proc or "").strip().lower()
    if "append" in p:
        return "appendectomy"
    if "chole" in p or "gall" in p:
        return "cholecystectomy"
    if "colect" in p or "colon" in p:
        return "colectomy"
    return p


def pod_bucket(pod: int) -> str:
    if pod <= 3:
        return "early"
    if pod <= 7:
        return "mid"
    return "late"


# =========================================================
# Ollama helper (optional)
# =========================================================
def ollama_generate(prompt: str, model: str = "mistral", timeout: int = 60) -> Optional[str]:
    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        return data.get("response")
    except Exception:
        return None


# =========================================================
# Pydantic State Schema
# =========================================================
class PipelineState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    raw_text: str = ""
    dataset_root: str = str(DEFAULT_DATASET_ROOT)
    faiss_cases_dir: str = str(DEFAULT_FAISS_CASES_DIR)
    faiss_guidelines_dir: str = str(DEFAULT_FAISS_GUIDELINES_DIR)
    top_k: int = 5
    use_ollama: bool = False
    rf_path: str = str(DEFAULT_RF_PATH)
    debug: bool = False

    pdf_name: str = ""
    run_mode: str = ""

    extracted_features: Dict[str, Any] = Field(default_factory=dict)
    validated_features: Dict[str, Any] = Field(default_factory=dict)
    validation: Dict[str, Any] = Field(default_factory=dict)
    retrieved_context: Dict[str, Any] = Field(default_factory=dict)
    temporal_analysis: Dict[str, Any] = Field(default_factory=dict)
    risk_assessment: Dict[str, Any] = Field(default_factory=dict)
    final_response: str = ""

    trace: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)


def _trace(state: PipelineState, msg: str) -> None:
    state.trace.append(msg)


def _langsmith_enabled() -> bool:
    v = os.getenv("LANGCHAIN_TRACING_V2", "").lower() in ("1", "true", "yes", "on")
    k = bool(os.getenv("LANGCHAIN_API_KEY"))
    return v and k


def merge_config(
    base: Optional[RunnableConfig],
    run_name: Optional[str] = None,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[RunnableConfig]:
    """Merge extra tags/metadata into an existing RunnableConfig (best-effort)."""
    if base is None and not _langsmith_enabled():
        return None

    out: Dict[str, Any] = {}
    if base:
        out.update(dict(base))

    if run_name:
        out["run_name"] = run_name

    # merge tags
    t: List[str] = []
    if out.get("tags"):
        t.extend(out["tags"])
    if tags:
        t.extend(tags)
    if t:
        seen = set()
        uniq = []
        for x in t:
            if x not in seen:
                uniq.append(x)
                seen.add(x)
        out["tags"] = uniq

    # merge metadata
    md: Dict[str, Any] = {}
    if out.get("metadata"):
        md.update(out["metadata"])
    if metadata:
        md.update(metadata)
    if md:
        out["metadata"] = md

    return out


# =========================================================
# LangChain Tools  (IMPORTANT: every @tool function has a docstring)
# =========================================================
@tool("extract_structured_features_regex")
def extract_structured_features_regex_tool(raw_text: str) -> Dict[str, Any]:
    """Regex-based structured feature extraction (fast, brittle)."""
    t = re.sub(r"[ \t]+", " ", raw_text or "")
    t = re.sub(r"\n+", "\n", t)

    def grab(pattern: str) -> Optional[str]:
        m = re.search(pattern, t, flags=re.IGNORECASE | re.DOTALL)
        return m.group(1).strip() if m else None

    features: Dict[str, Any] = {}
    features["patient_id"] = grab(r"Patient ID:\s*([A-Z]{2}\d{4})")
    pod_str = grab(r"Post-Op Day\s*(\d+)")
    features["pod"] = safe_int(pod_str) if pod_str else None
    features["procedure"] = grab(r"Procedure:\s*([^\n]+)")
    features["surgeon"] = grab(r"Surgeon:\s*([^\n]+)")
    features["timestamp"] = grab(r"Timestamp:\s*([^\n]+)")

    temp = grab(r"Temperature\s*([0-9]+(?:\.[0-9]+)?)")
    hr = grab(r"Heart Rate\s*(\d+)")
    bp = re.search(r"Blood Pressure\s*(\d{2,3})/(\d{2,3})", t, flags=re.IGNORECASE)
    rr = grab(r"Respiratory Rate\s*(\d+)")
    spo2 = grab(r"SpO2\s*(\d+)")
    pain = grab(r"Pain Score\s*(\d+)")

    features["temperature"] = safe_float(temp) if temp else None
    features["heart_rate"] = safe_int(hr) if hr else None
    if bp:
        features["bp_systolic"] = safe_int(bp.group(1))
        features["bp_diastolic"] = safe_int(bp.group(2))
    else:
        features["bp_systolic"] = None
        features["bp_diastolic"] = None
    features["respiratory_rate"] = safe_int(rr) if rr else None
    features["spo2"] = safe_int(spo2) if spo2 else None
    features["pain_score"] = safe_int(pain) if pain else None

    wbc = grab(r"\bWBC\s*([0-9]+(?:\.[0-9]+)?)")
    hgb = grab(r"Hemoglobin\s*([0-9]+(?:\.[0-9]+)?)")
    features["wbc"] = safe_float(wbc) if wbc else None
    features["hemoglobin"] = safe_float(hgb) if hgb else None

    platelets = grab(r"Platelets\s*([0-9]+(?:\.[0-9]+)?)")
    creatinine = grab(r"Creatinine\s*([0-9]+(?:\.[0-9]+)?)")
    features["platelets"] = safe_float(platelets) if platelets else None
    features["creatinine"] = safe_float(creatinine) if creatinine else None

    notes = grab(r"Clinical Notes\s*(.+?)Automated Risk")
    if not notes:
        notes = grab(r"Clinical Notes\s*(.+)$")
    features["clinical_notes"] = notes

    red_flags = re.findall(r"RED FLAG:\s*([A-Za-z \-]+)", t, flags=re.IGNORECASE)
    features["red_flags"] = [rf.strip() for rf in red_flags] if red_flags else []

    return features


@tool("extract_structured_features_spacy")
def extract_structured_features_spacy_tool(raw_text: str) -> Dict[str, Any]:
    """spaCy-based fallback extraction (local, no API key). Requires spaCy + en_core_web_sm."""
    if spacy is None:
        return {"_spacy_error": "spacy_not_installed"}

    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception as e:
        return {"_spacy_error": f"spacy_model_load_failed:{e}"}

    text = raw_text or ""
    doc = nlp(text)

    candidates = "\n".join([sent.text for sent in doc.sents])

    def find_num(label_patterns: List[str], cast="float") -> Optional[float]:
        for lp in label_patterns:
            m = re.search(lp + r"\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)?)", candidates, flags=re.IGNORECASE)
            if m:
                return safe_float(m.group(1)) if cast == "float" else safe_int(m.group(1))
        return None

    def find_bp() -> Tuple[Optional[int], Optional[int]]:
        m = re.search(r"(blood pressure|bp)\s*[:\-]?\s*(\d{2,3})/(\d{2,3})", candidates, flags=re.IGNORECASE)
        if not m:
            return None, None
        return safe_int(m.group(2)), safe_int(m.group(3))

    feats: Dict[str, Any] = {}
    m_pid = re.search(r"Patient ID:\s*([A-Z]{2}\d{4})", text, flags=re.IGNORECASE)
    feats["patient_id"] = m_pid.group(1).strip() if m_pid else None

    m_pod = re.search(r"(Post-Op Day|POD)\s*[:\-]?\s*(\d+)", text, flags=re.IGNORECASE)
    feats["pod"] = safe_int(m_pod.group(2)) if m_pod else None

    m_proc = re.search(r"Procedure:\s*([^\n]+)", text, flags=re.IGNORECASE)
    feats["procedure"] = m_proc.group(1).strip() if m_proc else None

    feats["temperature"] = find_num(["temperature", "temp"], cast="float")
    feats["heart_rate"] = find_num(["heart rate", r"\bhr\b"], cast="int")
    sbp, dbp = find_bp()
    feats["bp_systolic"], feats["bp_diastolic"] = sbp, dbp
    feats["respiratory_rate"] = find_num(["respiratory rate", r"\brr\b"], cast="int")
    feats["spo2"] = find_num(["spo2", r"\bO2 sat\b", r"\boxygen saturation\b"], cast="int")
    feats["pain_score"] = find_num(["pain score", r"\bpain\b"], cast="int")
    feats["wbc"] = find_num([r"\bwbc\b", "white blood cell"], cast="float")
    feats["hemoglobin"] = find_num(["hemoglobin", r"\bhgb\b"], cast="float")
    feats["platelets"] = find_num(["platelets", r"\bplt\b"], cast="float")
    feats["creatinine"] = find_num(["creatinine", r"\bcr\b"], cast="float")

    m_notes = re.search(r"Clinical Notes\s*(.+?)(Automated Risk|$)", text, flags=re.IGNORECASE | re.DOTALL)
    feats["clinical_notes"] = m_notes.group(1).strip() if m_notes else None
    red_flags = re.findall(r"RED FLAG:\s*([A-Za-z \-]+)", text, flags=re.IGNORECASE)
    feats["red_flags"] = [rf.strip() for rf in red_flags] if red_flags else []

    feats["_spacy_used"] = True
    return feats


@tool("extract_structured_features_ollama")
def extract_structured_features_ollama_tool(raw_text: str) -> Dict[str, Any]:
    """Ollama LLM extraction fallback (local). Returns JSON-parsed dict (best-effort)."""
    prompt = f"""
Extract clinical fields from this post-op note.
Return ONLY valid JSON with these keys:
patient_id, pod, procedure, surgeon, timestamp,
temperature, heart_rate, bp_systolic, bp_diastolic, respiratory_rate, spo2, pain_score,
wbc, hemoglobin, platelets, creatinine,
clinical_notes, red_flags

If value missing, set null. red_flags must be a list of strings.
Text:
{raw_text}
"""
    resp = ollama_generate(prompt, model="mistral", timeout=90)
    if not resp:
        return {"_ollama_error": "ollama_no_response"}

    try:
        m = re.search(r"\{.*\}", resp, flags=re.DOTALL)
        js = m.group(0) if m else resp
        data = json.loads(js)
        data["_ollama_used"] = True
        return data
    except Exception as e:
        return {"_ollama_error": f"ollama_json_parse_failed:{e}", "_ollama_raw": resp[:500]}


@tool("validate_features")
def validate_features_tool(features: Dict[str, Any]) -> Dict[str, Any]:
    """Validate required fields and physiological ranges; attach extraction_quality."""
    required = REQUIRED_FOR_QUALITY
    errors: List[str] = []
    warnings: List[str] = []

    for k in required:
        if features.get(k) is None:
            warnings.append(f"missing_required:{k}")

    for k, (lo, hi) in VALIDATION_RULES.items():
        v = features.get(k)
        if v is None:
            continue
        try:
            if not (lo <= float(v) <= hi):
                errors.append(f"out_of_range:{k}={v} not in [{lo},{hi}]")
        except Exception:
            errors.append(f"invalid_type:{k}={v}")

    status = "ok"
    if errors:
        status = "error"
    elif warnings:
        status = "warn"

    validated = dict(features)
    validated["validator_status"] = status
    validated["validator_errors"] = errors
    validated["validator_warnings"] = warnings
    validated["extraction_quality"] = quality_score(features, REQUIRED_FOR_QUALITY)

    return {
        "validation": {"status": status, "errors": errors, "warnings": warnings},
        "validated_features": validated,
    }


def craft_retrieval_query(vf: Dict[str, Any]) -> str:
    parts = []
    if vf.get("procedure"):
        parts.append(f"Procedure: {vf['procedure']}")
    if vf.get("pod") is not None:
        parts.append(f"Post-Op Day {vf['pod']}")
    for k in ["temperature", "heart_rate", "spo2", "wbc", "pain_score"]:
        if vf.get(k) is not None:
            parts.append(f"{k} {vf[k]}")
    if vf.get("red_flags"):
        parts.append("RED FLAG " + " ".join(vf["red_flags"]))
    if vf.get("clinical_notes"):
        parts.append(str(vf["clinical_notes"])[:200])
    return " | ".join(parts) if parts else "post-operative progress note"


def load_faiss_store(faiss_dir: Path):
    embeddings = HuggingFaceEmbeddings(model_name=MINILM_MODEL)
    return FAISS.load_local(str(faiss_dir), embeddings, allow_dangerous_deserialization=True)


def load_patient_history_from_jsons(dataset_root: Path, patient_id: str, current_pod: int, lookback: int = 7):
    json_dir = dataset_root / "jsons"
    if not json_dir.exists():
        return []

    history = []
    for jp in sorted(json_dir.glob("*.json")):
        pid, pod = parse_filename_for_pid_pod(jp.name)
        if pid != patient_id or pod is None:
            continue
        if pod <= current_pod and pod >= max(0, current_pod - lookback):
            try:
                data = json.loads(jp.read_text(encoding="utf-8"))
                history.append(data.get("daily_observation", {}))
            except Exception:
                pass
    history.sort(key=lambda x: x.get("pod", 0))
    return history


@tool("retrieve_context_expanded")
def retrieve_context_expanded_tool(
    validated_features: Dict[str, Any],
    dataset_root: str,
    faiss_cases_dir: str,
    faiss_guidelines_dir: str,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Retrieve patient_history (JSON sidecars), similar_cases (cases FAISS), and guidelines (guidelines FAISS)."""
    vf = validated_features or {}
    pid = vf.get("patient_id")
    pod = vf.get("pod")
    query = craft_retrieval_query(vf)

    patient_history = []
    if pid and pod is not None:
        patient_history = load_patient_history_from_jsons(Path(dataset_root).resolve(), str(pid), int(pod), lookback=7)

    # similar cases
    cases_ctx = {"query_used": query, "top_k": 0, "citations": [], "contexts": [], "warning": None}
    cases_path = Path(faiss_cases_dir).resolve()
    if cases_path.exists():
        store = load_faiss_store(cases_path)
        retriever = store.as_retriever(search_kwargs={"k": int(top_k)})
        docs = retriever.invoke(query) if hasattr(retriever, "invoke") else retriever.get_relevant_documents(query)

        contexts = []
        citations = []
        for d in docs:
            meta = d.metadata or {}
            src = meta.get("source_file") or meta.get("source") or "unknown"
            page = meta.get("page")
            contexts.append({"source": src, "page": page, "text": d.page_content})
            citations.append(f"{src}" + (f":p{page}" if page is not None else ""))
        cases_ctx = {"query_used": query, "top_k": int(top_k), "citations": citations, "contexts": contexts, "warning": None}
    else:
        cases_ctx["warning"] = f"faiss_cases_dir_missing:{cases_path}"

    # guidelines
    guidelines_ctx = {"query_used": query, "top_k": 0, "citations": [], "contexts": [], "warning": None}
    guidelines_path = Path(faiss_guidelines_dir).resolve()
    if guidelines_path.exists():
        store_g = load_faiss_store(guidelines_path)
        retriever_g = store_g.as_retriever(search_kwargs={"k": int(top_k)})
        docs_g = retriever_g.invoke(query) if hasattr(retriever_g, "invoke") else retriever_g.get_relevant_documents(query)

        contexts_g = []
        citations_g = []
        for d in docs_g:
            meta = d.metadata or {}
            src = meta.get("source_file") or meta.get("source") or "guideline_unknown"
            page = meta.get("page")
            contexts_g.append({"source": src, "page": page, "text": d.page_content})
            citations_g.append(f"{src}" + (f":p{page}" if page is not None else ""))
        guidelines_ctx = {"query_used": query, "top_k": int(top_k), "citations": citations_g, "contexts": contexts_g, "warning": None}
    else:
        guidelines_ctx["warning"] = f"faiss_guidelines_dir_missing:{guidelines_path}"

    return {"query_used": query, "patient_history": patient_history, "similar_cases": cases_ctx, "guidelines": guidelines_ctx}


def trend(values: List[float]) -> str:
    if len(values) < 2:
        return "insufficient_history"
    if values[-1] > values[0] + 0.2:
        return "worsening"
    if values[-1] < values[0] - 0.2:
        return "improving"
    return "stable"


def rapid_deterioration(series: List[float], delta_threshold: float) -> bool:
    """True if last-step increase is >= threshold (best-effort)."""
    if len(series) < 2:
        return False
    return (series[-1] - series[-2]) >= delta_threshold


@tool("temporal_analysis_upgraded")
def temporal_analysis_upgraded_tool(validated_features: Dict[str, Any], retrieved_context: Dict[str, Any]) -> Dict[str, Any]:
    """Temporal trends + expected POD ranges + rapid deterioration flags using patient_history."""
    vf = validated_features or {}
    pid = vf.get("patient_id")
    pod = vf.get("pod")
    proc_norm = normalize_procedure(vf.get("procedure"))

    history = (retrieved_context or {}).get("patient_history") or []

    def extract_series(key: str) -> List[float]:
        vals: List[float] = []
        for h in history:
            v = h.get(key)
            if v is not None:
                try:
                    vals.append(float(v))
                except Exception:
                    pass
        cv = vf.get(key)
        if cv is not None:
            try:
                vals.append(float(cv))
            except Exception:
                pass
        return vals

    if not pid or pod is None:
        return {"status": "warn", "reason": "missing patient_id/pod", "history_points_used": len(history)}

    bucket = pod_bucket(int(pod))
    ranges = EXPECTED_RANGES_BY_PROCEDURE.get(proc_norm) or EXPECTED_RANGES_DEFAULT
    exp = ranges.get(bucket, EXPECTED_RANGES_DEFAULT["mid"])

    deviations = []
    for k, (lo, hi) in exp.items():
        v = vf.get(k)
        if v is None:
            continue
        try:
            fv = float(v)
            if fv < lo or fv > hi:
                deviations.append(
                    f"expected_range_violation:{k}={fv} not_in [{lo},{hi}] (bucket={bucket}, proc={proc_norm or 'unknown'})"
                )
        except Exception:
            continue

    temp_s = extract_series("temperature")
    hr_s = extract_series("heart_rate")
    wbc_s = extract_series("wbc")

    rapid = {
        "temp_rapid_rise": rapid_deterioration(temp_s, 0.7),
        "hr_rapid_rise": rapid_deterioration(hr_s, 20.0),
        "wbc_rapid_rise": rapid_deterioration(wbc_s, 2.0),
    }
    rapid_any = any(rapid.values())

    return {
        "status": "ok" if history else "warn",
        "history_points_used": len(history),
        "temperature_trend": trend(temp_s),
        "heart_rate_trend": trend(hr_s),
        "wbc_trend": trend(wbc_s),
        "spo2_trend": trend(extract_series("spo2")),
        "expected_ranges_bucket": bucket,
        "expected_ranges_used": exp,
        "deviations": deviations,
        "rapid_deterioration": rapid,
        "rapid_deterioration_any": rapid_any,
    }


def rule_risk_score(vf: Dict[str, Any], temporal: Dict[str, Any]) -> Tuple[int, str, List[str]]:
    score = 0
    factors: List[str] = []

    temp = vf.get("temperature") or 0
    hr = vf.get("heart_rate") or 0
    bps = vf.get("bp_systolic") or 120
    spo2 = vf.get("spo2") or 100
    wbc = vf.get("wbc") or 7
    pain = vf.get("pain_score") or 0
    notes = (vf.get("clinical_notes") or "").lower()
    red_flags = [str(rf).lower() for rf in (vf.get("red_flags") or [])]

    if temp > 39.0:
        score += 30
        factors.append("temp>39")
    elif temp > 38.0:
        score += 15
        factors.append("temp>38")

    if hr > 120:
        score += 25
        factors.append("hr>120")
    elif hr > 100:
        score += 15
        factors.append("hr>100")

    if bps < 90:
        score += 25
        factors.append("sbp<90")

    if spo2 < 90:
        score += 30
        factors.append("spo2<90")
    elif spo2 < 95:
        score += 15
        factors.append("spo2<95")

    if wbc > 15.0:
        score += 20
        factors.append("wbc>15")
    elif wbc > 11.0:
        score += 10
        factors.append("wbc>11")

    if pain > 7:
        score += 15
        factors.append("pain>7")
    elif pain > 5:
        score += 5
        factors.append("pain>5")

    for flag in ["sepsis", "wound dehiscence", "pulmonary embolism", "hemorrhage", "shock"]:
        if flag in notes or flag in " ".join(red_flags):
            score += 40
            factors.append(f"red_flag:{flag}")

    if temporal.get("rapid_deterioration_any"):
        score += 15
        factors.append("rapid_deterioration_any")

    deviations = temporal.get("deviations") or []
    if deviations:
        score += min(10, 2 * len(deviations))
        factors.append(f"expected_range_deviations:{len(deviations)}")

    if score >= 76:
        level = "CRITICAL"
    elif score >= 51:
        level = "HIGH"
    elif score >= 26:
        level = "MEDIUM"
    else:
        level = "LOW"
    return int(score), level, factors


def ml_features_vector(vf: Dict[str, Any], temporal: Dict[str, Any]) -> np.ndarray:
    def tcode(x: str) -> int:
        return {"improving": -1, "stable": 0, "worsening": 1}.get(x, 0)

    vec = [
        float(vf.get("temperature") or 0),
        float(vf.get("heart_rate") or 0),
        float(vf.get("bp_systolic") or 0),
        float(vf.get("spo2") or 0),
        float(vf.get("wbc") or 0),
        float(vf.get("hemoglobin") or 0),
        float(vf.get("pain_score") or 0),
        float(vf.get("pod") or 0),
        tcode((temporal or {}).get("temperature_trend", "stable")),
        tcode((temporal or {}).get("wbc_trend", "stable")),
        len(vf.get("red_flags") or []),
    ]
    return np.array(vec, dtype=np.float32).reshape(1, -1)


@tool("risk_assessment")
def risk_assessment_tool(
    validated_features: Dict[str, Any],
    temporal_analysis: Dict[str, Any],
    retrieved_context: Dict[str, Any],
    rf_path: str,
    use_ollama: bool = False,
) -> Dict[str, Any]:
    """Compute risk using rules + optional RF + optional Ollama reasoning."""
    vf = validated_features or {}
    temporal = temporal_analysis or {}
    retrieved = retrieved_context or {}

    rule_score, rule_level, rule_factors = rule_risk_score(vf, temporal)

    ml_pred = None
    ml_proba = None
    rf_file = Path(rf_path).resolve()
    if rf_file.exists():
        try:
            rf = joblib.load(rf_file)
            X = ml_features_vector(vf, temporal)
            ml_pred = rf.predict(X)[0]
            if hasattr(rf, "predict_proba"):
                probs = rf.predict_proba(X)[0]
                ml_proba = {str(rf.classes_[i]): float(probs[i]) for i in range(len(rf.classes_))}
        except Exception:
            ml_pred = None
            ml_proba = None

    llm_reasoning = None
    if use_ollama:
        prompt = f"""
You are a clinical risk assistant for post-op monitoring.
Given:
- validated_features: {json.dumps(vf, indent=2)}
- temporal_analysis: {json.dumps(temporal, indent=2)}
- similar_cases_citations: {(retrieved.get('similar_cases') or {}).get('citations', [])}
- guideline_citations: {(retrieved.get('guidelines') or {}).get('citations', [])}

Produce:
1) concise risk reasoning (2-4 bullet points)
2) likely complications to consider
3) any urgent alerts
Return plain text.
"""
        llm_reasoning = ollama_generate(prompt, model="mistral", timeout=60)

    final_level = rule_level
    final_score = rule_score
    factors = list(rule_factors)

    if ml_pred in RISK_TIERS and RISK_TIERS.index(ml_pred) > RISK_TIERS.index(final_level):
        final_level = ml_pred
        factors.append(f"ml_escalation:{ml_pred}")

    alerts: List[str] = []
    if final_level in ["HIGH", "CRITICAL"]:
        alerts.append("Urgent clinical review recommended")
    if any("red_flag:" in f for f in factors):
        alerts.append("RED FLAG detected in notes")
    if temporal.get("rapid_deterioration_any"):
        alerts.append("Rapid deterioration detected (trend-based)")

    return {
        "risk_level": final_level,
        "risk_score": final_score,
        "alerts": alerts,
        "rule_score": rule_score,
        "rule_level": rule_level,
        "rule_factors": rule_factors,
        "ml_pred": ml_pred,
        "ml_proba": ml_proba,
        "final_factors": factors,
        "llm_reasoning": llm_reasoning,
    }


def template_response(
    level: str,
    vf: Dict[str, Any],
    temporal: Dict[str, Any],
    risk: Dict[str, Any],
    citations: List[str],
    guideline_citations: List[str],
) -> str:
    alerts = risk.get("alerts", [])
    pid = vf.get("patient_id", "UNKNOWN")
    pod = vf.get("pod", "NA")
    proc = vf.get("procedure", "NA")

    lines: List[str] = []
    lines.append(f"Clinical Assessment (Synthetic Demo) — Patient {pid} | POD {pod} | Procedure: {proc}")

    lines.append("\nSummary:")
    lines.append(f"- Risk Tier: {level} (risk_score={risk.get('risk_score')}, ml_pred={risk.get('ml_pred')})")
    lines.append(
        f"- Vitals: Temp={vf.get('temperature')} HR={vf.get('heart_rate')} SBP={vf.get('bp_systolic')} SpO2={vf.get('spo2')}"
    )
    lines.append(f"- Labs: WBC={vf.get('wbc')} Hgb={vf.get('hemoglobin')} Pain={vf.get('pain_score')}")

    lines.append("\nTrends & Checks:")
    lines.append(f"- Temp trend: {temporal.get('temperature_trend')}")
    lines.append(f"- WBC trend:  {temporal.get('wbc_trend')}")
    lines.append(f"- HR trend:   {temporal.get('heart_rate_trend')}")
    lines.append(f"- SpO2 trend: {temporal.get('spo2_trend')}")
    if temporal.get("deviations"):
        lines.append(f"- Expected-range deviations: {len(temporal.get('deviations') or [])}")
    if temporal.get("rapid_deterioration_any"):
        lines.append(f"- Rapid deterioration: {temporal.get('rapid_deterioration')}")

    lines.append("\nFactors:")
    ff = risk.get("final_factors") or risk.get("rule_factors") or []
    if not ff:
        lines.append("- (none)")
    else:
        for f in ff[:10]:
            lines.append(f"- {f}")

    if alerts:
        lines.append("\nALERTS:")
        for a in alerts:
            lines.append(f"- {a}")

    lines.append("\nRecommendations:")
    if level == "LOW":
        lines.append("- Continue routine post-op monitoring and reassess as scheduled.")
    elif level == "MEDIUM":
        lines.append("- Repeat vitals/labs; reassess within 4–6 hours. Consider early infection workup if symptoms evolve.")
    elif level == "HIGH":
        lines.append("- Urgent clinician review. Consider cultures/imaging; screen for sepsis if suspected.")
    else:
        lines.append("- Immediate escalation. Activate emergency pathway (sepsis/PE/hemorrhage consideration).")

    if citations:
        lines.append("\nRetrieved similar-case citations:")
        for c in citations[:8]:
            lines.append(f"- {c}")

    if guideline_citations:
        lines.append("\nRetrieved guideline citations:")
        for c in guideline_citations[:8]:
            lines.append(f"- {c}")

    lines.append("\nDisclaimer: Synthetic data for testing only. Not medical advice.")
    return "\n".join(lines)


@tool("generate_response")
def generate_response_tool(
    validated_features: Dict[str, Any],
    temporal_analysis: Dict[str, Any],
    risk_assessment: Dict[str, Any],
    retrieved_context: Dict[str, Any],
    use_ollama: bool = False,
    response_style: str = "default",
) -> str:
    """Generate clinician-friendly response from templates + optional Ollama polishing."""
    vf = validated_features or {}
    temporal = temporal_analysis or {}
    risk = risk_assessment or {}

    sim = (retrieved_context or {}).get("similar_cases") or {}
    gl = (retrieved_context or {}).get("guidelines") or {}
    citations = sim.get("citations") or []
    guideline_citations = gl.get("citations") or []

    level = risk.get("risk_level", "MEDIUM")
    base = template_response(level, vf, temporal, risk, citations, guideline_citations)

    if use_ollama:
        prompt = f"""
Rewrite the following assessment to be concise and clinician-friendly.
Do NOT add new facts. Keep the same risk tier and recommendations.
Text:
{base}
"""
        polished = ollama_generate(prompt, model="mistral", timeout=60)
        if polished and len(polished.strip()) > 50:
            return polished.strip()

    return base


# =========================================================
# LangGraph Nodes
# =========================================================
def agent_parser(state: PipelineState, config: Optional[RunnableConfig] = None) -> PipelineState:
    _trace(state, "agent_parser:start")

    cfg = merge_config(config, run_name="parser.regex", tags=["agent:parser", "method:regex"], metadata={"pdf": state.pdf_name})
    feats = extract_structured_features_regex_tool.invoke({"raw_text": state.raw_text}, config=cfg)
    q = quality_score(feats, REQUIRED_FOR_QUALITY)
    feats["_extraction_quality"] = q
    feats["_extraction_method"] = "regex"

    if q < QUALITY_THRESHOLD:
        _trace(state, f"agent_parser:fallback_to_spacy quality={q:.2f}")
        cfg2 = merge_config(
            config,
            run_name="parser.spacy",
            tags=["agent:parser", "method:spacy"],
            metadata={"pdf": state.pdf_name, "prior_quality": q},
        )
        sp = extract_structured_features_spacy_tool.invoke({"raw_text": state.raw_text}, config=cfg2)

        merged = dict(feats)
        for k, v in sp.items():
            if merged.get(k) is None and v is not None:
                merged[k] = v
        merged["_extraction_method"] = "regex+spacy"
        merged["_spacy_status"] = sp.get("_spacy_error") or True

        feats = merged
        q = quality_score(feats, REQUIRED_FOR_QUALITY)
        feats["_extraction_quality"] = q

    if q < QUALITY_THRESHOLD and state.use_ollama:
        _trace(state, f"agent_parser:fallback_to_ollama quality={q:.2f}")
        cfg3 = merge_config(
            config,
            run_name="parser.ollama",
            tags=["agent:parser", "method:ollama"],
            metadata={"pdf": state.pdf_name, "prior_quality": q},
        )
        ol = extract_structured_features_ollama_tool.invoke({"raw_text": state.raw_text}, config=cfg3)

        merged = dict(feats)
        for k, v in ol.items():
            if merged.get(k) is None and v is not None:
                merged[k] = v
        merged["_extraction_method"] = "regex+spacy+ollama" if "spacy" in merged.get("_extraction_method", "") else "regex+ollama"
        merged["_ollama_status"] = ol.get("_ollama_error") or True

        feats = merged
        q = quality_score(feats, REQUIRED_FOR_QUALITY)
        feats["_extraction_quality"] = q

    feats["_manual_review_recommended"] = bool(q < QUALITY_THRESHOLD)
    state.extracted_features = feats

    if feats.get("_manual_review_recommended"):
        state.errors.append(f"parser_low_quality:{q:.2f}")

    _trace(state, "agent_parser:done")
    return state


def agent_validator(state: PipelineState, config: Optional[RunnableConfig] = None) -> PipelineState:
    _trace(state, "agent_validator:start")
    pid = (state.extracted_features or {}).get("patient_id")
    pod = (state.extracted_features or {}).get("pod")
    cfg = merge_config(config, run_name="validator", tags=["agent:validator"], metadata={"patient_id": pid, "pod": pod, "pdf": state.pdf_name})
    out = validate_features_tool.invoke({"features": state.extracted_features}, config=cfg)
    state.validation = out.get("validation", {})
    state.validated_features = out.get("validated_features", {})
    _trace(state, "agent_validator:done")
    return state


def agent_retrieval(state: PipelineState, config: Optional[RunnableConfig] = None) -> PipelineState:
    _trace(state, "agent_retrieval:start")
    pid = (state.validated_features or {}).get("patient_id")
    pod = (state.validated_features or {}).get("pod")
    cfg = merge_config(config, run_name="retrieval", tags=["agent:retrieval"], metadata={"patient_id": pid, "pod": pod, "pdf": state.pdf_name})
    rc = retrieve_context_expanded_tool.invoke(
        {
            "validated_features": state.validated_features,
            "dataset_root": state.dataset_root,
            "faiss_cases_dir": state.faiss_cases_dir,
            "faiss_guidelines_dir": state.faiss_guidelines_dir,
            "top_k": state.top_k,
        },
        config=cfg,
    )
    state.retrieved_context = rc

    sim_warn = (rc.get("similar_cases") or {}).get("warning")
    gl_warn = (rc.get("guidelines") or {}).get("warning")
    if sim_warn:
        state.errors.append(f"retrieval_warning:{sim_warn}")
    if gl_warn:
        state.errors.append(f"retrieval_warning:{gl_warn}")

    _trace(state, "agent_retrieval:done")
    return state


def agent_temporal(state: PipelineState, config: Optional[RunnableConfig] = None) -> PipelineState:
    _trace(state, "agent_temporal:start")
    pid = (state.validated_features or {}).get("patient_id")
    pod = (state.validated_features or {}).get("pod")
    cfg = merge_config(config, run_name="temporal", tags=["agent:temporal"], metadata={"patient_id": pid, "pod": pod, "pdf": state.pdf_name})
    ta = temporal_analysis_upgraded_tool.invoke({"validated_features": state.validated_features, "retrieved_context": state.retrieved_context}, config=cfg)
    state.temporal_analysis = ta
    _trace(state, "agent_temporal:done")
    return state


def agent_risk_assessor(state: PipelineState, config: Optional[RunnableConfig] = None) -> PipelineState:
    _trace(state, "agent_risk_assessor:start")
    pid = (state.validated_features or {}).get("patient_id")
    pod = (state.validated_features or {}).get("pod")
    cfg = merge_config(config, run_name="risk", tags=["agent:risk"], metadata={"patient_id": pid, "pod": pod, "pdf": state.pdf_name})
    try:
        ra = risk_assessment_tool.invoke(
            {
                "validated_features": state.validated_features,
                "temporal_analysis": state.temporal_analysis,
                "retrieved_context": state.retrieved_context,
                "rf_path": state.rf_path,
                "use_ollama": state.use_ollama,
            },
            config=cfg,
        )
        state.risk_assessment = ra
    except Exception as e:
        state.errors.append(f"agent_risk_assessor_failed:{e}")
        state.risk_assessment = {"risk_level": "MEDIUM", "risk_score": 0, "alerts": ["Risk engine failed; defaulting to MEDIUM"], "final_factors": []}
    _trace(state, "agent_risk_assessor:done")
    return state


def agent_respond_low(state: PipelineState, config: Optional[RunnableConfig] = None) -> PipelineState:
    _trace(state, "agent_respond_low:start")
    pid = (state.validated_features or {}).get("patient_id")
    pod = (state.validated_features or {}).get("pod")
    rl = (state.risk_assessment or {}).get("risk_level")
    cfg = merge_config(config, run_name="respond.low", tags=["agent:respond", "tier:low"], metadata={"patient_id": pid, "pod": pod, "risk_level": rl, "pdf": state.pdf_name})
    state.final_response = generate_response_tool.invoke(
        {
            "validated_features": state.validated_features,
            "temporal_analysis": state.temporal_analysis,
            "risk_assessment": state.risk_assessment,
            "retrieved_context": state.retrieved_context,
            "use_ollama": state.use_ollama,
            "response_style": "low",
        },
        config=cfg,
    )
    _trace(state, "agent_respond_low:done")
    return state


def agent_respond_medium(state: PipelineState, config: Optional[RunnableConfig] = None) -> PipelineState:
    _trace(state, "agent_respond_medium:start")
    pid = (state.validated_features or {}).get("patient_id")
    pod = (state.validated_features or {}).get("pod")
    rl = (state.risk_assessment or {}).get("risk_level")
    cfg = merge_config(config, run_name="respond.medium", tags=["agent:respond", "tier:medium"], metadata={"patient_id": pid, "pod": pod, "risk_level": rl, "pdf": state.pdf_name})
    state.final_response = generate_response_tool.invoke(
        {
            "validated_features": state.validated_features,
            "temporal_analysis": state.temporal_analysis,
            "risk_assessment": state.risk_assessment,
            "retrieved_context": state.retrieved_context,
            "use_ollama": state.use_ollama,
            "response_style": "medium",
        },
        config=cfg,
    )
    _trace(state, "agent_respond_medium:done")
    return state


def agent_respond_high(state: PipelineState, config: Optional[RunnableConfig] = None) -> PipelineState:
    _trace(state, "agent_respond_high:start")
    pid = (state.validated_features or {}).get("patient_id")
    pod = (state.validated_features or {}).get("pod")
    rl = (state.risk_assessment or {}).get("risk_level")
    cfg = merge_config(config, run_name="respond.high", tags=["agent:respond", "tier:high"], metadata={"patient_id": pid, "pod": pod, "risk_level": rl, "pdf": state.pdf_name})
    state.final_response = generate_response_tool.invoke(
        {
            "validated_features": state.validated_features,
            "temporal_analysis": state.temporal_analysis,
            "risk_assessment": state.risk_assessment,
            "retrieved_context": state.retrieved_context,
            "use_ollama": state.use_ollama,
            "response_style": "high",
        },
        config=cfg,
    )
    _trace(state, "agent_respond_high:done")
    return state


def agent_respond_critical(state: PipelineState, config: Optional[RunnableConfig] = None) -> PipelineState:
    _trace(state, "agent_respond_critical:start")
    pid = (state.validated_features or {}).get("patient_id")
    pod = (state.validated_features or {}).get("pod")
    rl = (state.risk_assessment or {}).get("risk_level")
    cfg = merge_config(config, run_name="respond.critical", tags=["agent:respond", "tier:critical"], metadata={"patient_id": pid, "pod": pod, "risk_level": rl, "pdf": state.pdf_name})
    state.final_response = generate_response_tool.invoke(
        {
            "validated_features": state.validated_features,
            "temporal_analysis": state.temporal_analysis,
            "risk_assessment": state.risk_assessment,
            "retrieved_context": state.retrieved_context,
            "use_ollama": state.use_ollama,
            "response_style": "critical",
        },
        config=cfg,
    )
    _trace(state, "agent_respond_critical:done")
    return state


def route_after_risk(state: PipelineState) -> str:
    level = (state.risk_assessment or {}).get("risk_level", "MEDIUM")
    if level == "LOW":
        return "respond_low"
    if level == "MEDIUM":
        return "respond_medium"
    if level == "HIGH":
        return "respond_high"
    return "respond_critical"


def build_graph():
    g = StateGraph(PipelineState)

    g.add_node("parse", agent_parser)
    g.add_node("validate", agent_validator)
    g.add_node("retrieve", agent_retrieval)
    g.add_node("temporal", agent_temporal)
    g.add_node("assess_risk", agent_risk_assessor)

    g.add_node("respond_low", agent_respond_low)
    g.add_node("respond_medium", agent_respond_medium)
    g.add_node("respond_high", agent_respond_high)
    g.add_node("respond_critical", agent_respond_critical)

    g.set_entry_point("parse")
    g.add_edge("parse", "validate")
    g.add_edge("validate", "retrieve")
    g.add_edge("retrieve", "temporal")
    g.add_edge("temporal", "assess_risk")

    g.add_conditional_edges(
        "assess_risk",
        route_after_risk,
        {
            "respond_low": "respond_low",
            "respond_medium": "respond_medium",
            "respond_high": "respond_high",
            "respond_critical": "respond_critical",
        },
    )

    g.add_edge("respond_low", END)
    g.add_edge("respond_medium", END)
    g.add_edge("respond_high", END)
    g.add_edge("respond_critical", END)

    return g.compile()


# =========================================================
# ML training
# =========================================================
def train_random_forest(dataset_root: Path, out_path: Path):
    json_dir = dataset_root / "jsons"
    if not json_dir.exists():
        raise FileNotFoundError(f"Missing json dir: {json_dir}")

    X_list = []
    y_list = []

    for jp in json_dir.glob("*.json"):
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
            dob = data.get("daily_observation", {})
            gt = data.get("ground_truth", {})

            vf = {
                "temperature": dob.get("temperature"),
                "heart_rate": dob.get("heart_rate"),
                "bp_systolic": safe_int(str(dob.get("blood_pressure", "120/80")).split("/")[0]) if dob.get("blood_pressure") else 120,
                "spo2": dob.get("spo2"),
                "wbc": dob.get("wbc"),
                "hemoglobin": dob.get("hemoglobin"),
                "pain_score": dob.get("pain_score"),
                "pod": dob.get("pod"),
                "red_flags": re.findall(r"RED FLAG:\s*([A-Za-z \-]+)", (dob.get("clinical_notes") or ""), flags=re.IGNORECASE),
            }

            temporal = {"temperature_trend": "stable", "wbc_trend": "stable"}
            X = ml_features_vector(vf, temporal).reshape(-1)
            y = gt.get("risk_level_rule") or dob.get("rule_risk_level")

            if y not in RISK_TIERS:
                continue

            X_list.append(X)
            y_list.append(y)
        except Exception:
            continue

    if len(X_list) < 20:
        raise RuntimeError(f"Not enough training samples found in {json_dir} (found {len(X_list)})")

    X_arr = np.vstack(X_list)
    y_arr = np.array(y_list)

    rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    rf.fit(X_arr, y_arr)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf, out_path)
    print(f"Trained RandomForest on {len(X_arr)} samples")
    print(f"Saved model to: {out_path}")


# =========================================================
# Run helpers
# =========================================================
def make_run_config(run_name: str, tags: List[str], metadata: Dict[str, Any]) -> RunnableConfig:
    return {"run_name": run_name, "tags": tags, "metadata": metadata}


def run_one_pdf(graph, pdf_path: Path, args) -> PipelineState:
    raw_text = read_pdf_text(pdf_path)

    init_state = PipelineState(
        raw_text=raw_text,
        dataset_root=str(Path(args.dataset_root).resolve()),
        faiss_cases_dir=str(Path(args.faiss_cases_dir).resolve()),
        faiss_guidelines_dir=str(Path(args.faiss_guidelines_dir).resolve()),
        top_k=int(args.top_k),
        use_ollama=bool(args.use_ollama),
        rf_path=str(Path(args.rf_path).resolve()) if args.rf_path else str(DEFAULT_RF_PATH),
        debug=bool(args.debug),
        pdf_name=pdf_path.name,
        run_mode=args.mode,
    )

    pid, pod = parse_filename_for_pid_pod(pdf_path.name)
    run_name = f"{pid or 'UNKNOWN'}_POD{pod if pod is not None else 'NA'}__{pdf_path.stem}"
    tags = [args.mode, "medicalRAG", "clinical-risk"]
    metadata = {
        "pdf": pdf_path.name,
        "patient_id_guess": pid,
        "pod_guess": pod,
        "faiss_cases_dir": str(Path(args.faiss_cases_dir).resolve()),
        "faiss_guidelines_dir": str(Path(args.faiss_guidelines_dir).resolve()),
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "top_k": int(args.top_k),
        "use_ollama": bool(args.use_ollama),
    }
    config = make_run_config(run_name, tags, metadata) if _langsmith_enabled() else None

    out = graph.invoke(init_state, config=config)

    if isinstance(out, PipelineState):
        return out
    if isinstance(out, dict):
        return PipelineState(**out)
    return PipelineState()


def run_folder(args) -> None:
    pdf_dir = Path(args.pdf_dir).resolve()
    if not pdf_dir.exists():
        raise FileNotFoundError(f"--pdf-dir not found: {pdf_dir}")

    out_dir = Path(args.out_dir).resolve()
    ensure_dir(out_dir)
    jsonl_path = out_dir / "results.jsonl"
    csv_path = out_dir / "results.csv"

    graph = build_graph()
    pdfs = sorted(list(pdf_dir.glob("*.pdf")))
    if not pdfs:
        raise RuntimeError(f"No PDFs found in {pdf_dir}")

    results: List[Dict[str, Any]] = []
    alerts_all: List[str] = []

    with jsonl_path.open("w", encoding="utf-8") as jf:
        for pdf in pdfs:
            st = run_one_pdf(graph, pdf, args)

            vf = st.validated_features or {}
            risk = st.risk_assessment or {}
            rc = st.retrieved_context or {}
            sim = (rc.get("similar_cases") or {})
            gl = (rc.get("guidelines") or {})

            row = {
                "pdf": pdf.name,
                "patient_id": vf.get("patient_id"),
                "pod": vf.get("pod"),
                "procedure": vf.get("procedure"),
                "risk_level": risk.get("risk_level"),
                "risk_score": risk.get("risk_score"),
                "alerts": risk.get("alerts") or [],
                "citations_similar_cases": sim.get("citations") or [],
                "citations_guidelines": gl.get("citations") or [],
                "retrieval_query_used": rc.get("query_used"),
                "parser_quality": vf.get("extraction_quality"),
                "errors": st.errors,
            }
            results.append(row)
            jf.write(json.dumps(row) + "\n")

            for a in (row["alerts"] or []):
                alerts_all.append(f"{pdf.name} | {row.get('patient_id')} | {row.get('risk_level')} | {a}")

    with csv_path.open("w", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(
            cf,
            fieldnames=[
                "pdf",
                "patient_id",
                "pod",
                "procedure",
                "risk_level",
                "risk_score",
                "alerts",
                "citations_similar_cases",
                "citations_guidelines",
                "retrieval_query_used",
                "parser_quality",
                "errors",
            ],
        )
        writer.writeheader()
        for r in results:
            r2 = dict(r)
            r2["alerts"] = "; ".join(r2.get("alerts") or [])
            r2["citations_similar_cases"] = "; ".join(r2.get("citations_similar_cases") or [])
            r2["citations_guidelines"] = "; ".join(r2.get("citations_guidelines") or [])
            r2["errors"] = "; ".join(r2.get("errors") or [])
            writer.writerow(r2)

    counts = summarize_counts(results)

    print("\n================ FOLDER SUMMARY ================\n")
    print(f"Processed PDFs: {len(results)}")
    print("Risk tier counts:")
    for k in RISK_TIERS:
        print(f"  {k:8s}: {counts.get(k, 0)}")

    if alerts_all:
        print("\nAlerts:")
        for line in alerts_all[:50]:
            print(" -", line)
        if len(alerts_all) > 50:
            print(f" ... ({len(alerts_all) - 50} more)")
    else:
        print("\nAlerts: (none)")

    print(f"\nSaved JSONL: {jsonl_path}")
    print(f"Saved CSV:   {csv_path}")
    print("\n===============================================\n")


# =========================================================
# CLI
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["train_ml", "run_pdf", "run_folder"])

    parser.add_argument("--pdf", type=str, default=None)
    parser.add_argument("--pdf-dir", type=str, default=None, help="Folder of PDFs (for mode=run_folder)")
    parser.add_argument("--out-dir", type=str, default="results", help="Output folder for mode=run_folder")

    parser.add_argument("--dataset-root", type=str, default=str(DEFAULT_DATASET_ROOT))

    parser.add_argument("--faiss-cases-dir", type=str, default=str(DEFAULT_FAISS_CASES_DIR))
    parser.add_argument("--faiss-guidelines-dir", type=str, default=str(DEFAULT_FAISS_GUIDELINES_DIR))

    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--use-ollama", action="store_true")
    parser.add_argument("--rf-path", type=str, default=str(DEFAULT_RF_PATH))
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    dataset_root = Path(args.dataset_root).resolve()

    if args.mode == "train_ml":
        train_random_forest(dataset_root, Path(args.rf_path).resolve())
        return

    if args.mode == "run_pdf":
        if not args.pdf:
            raise ValueError("--pdf is required for mode=run_pdf")
        pdf_path = Path(args.pdf).resolve()
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        graph = build_graph()
        st = run_one_pdf(graph, pdf_path, args)

        if args.debug:
            print("\n---- DEBUG TRACE ----")
            print(st.trace)
            print("\n---- DEBUG ERRORS ----")
            print(st.errors)

        print("\n================ FINAL OUTPUT ================\n")
        print(st.final_response or "NO RESPONSE")
        print("\n=============================================\n")

        if args.debug:
            print("validation:", json.dumps(st.validation, indent=2))
            print("risk:", json.dumps(st.risk_assessment, indent=2))
            print("retrieval_query_used:", (st.retrieved_context or {}).get("query_used"))
            sim = (st.retrieved_context or {}).get("similar_cases") or {}
            gl = (st.retrieved_context or {}).get("guidelines") or {}
            print("citations_similar_cases:", sim.get("citations"))
            print("citations_guidelines:", gl.get("citations"))
            print("temporal:", json.dumps(st.temporal_analysis, indent=2))
        return

    if args.mode == "run_folder":
        if not args.pdf_dir:
            raise ValueError("--pdf-dir is required for mode=run_folder")
        run_folder(args)
        return


if __name__ == "__main__":
    main()


