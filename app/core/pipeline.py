"""
core/pipeline.py — Thin wrapper around clinical_langgraph_pipeline.py.

Exposes run_single_pdf() and run_pdf_batch() as clean Python functions
so the FastAPI routers don't need to import internals directly.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

# ── Make sure the repo root is importable ──────────────────────────────────
# Adjust this path if your scripts/ folder is elsewhere relative to app/
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.clinical_langgraph_pipeline import (   # noqa: E402
    PipelineState,
    build_graph,
    run_one_pdf,
    DEFAULT_FAISS_CASES_DIR,
    DEFAULT_FAISS_GUIDELINES_DIR,
    DEFAULT_DATASET_ROOT,
    DEFAULT_RF_PATH,
)


# ── Lightweight args shim (mirrors argparse Namespace) ─────────────────────
class _Args:
    def __init__(
        self,
        dataset_root: str,
        faiss_cases_dir: str,
        faiss_guidelines_dir: str,
        top_k: int,
        use_ollama: bool,
        rf_path: str,
        debug: bool,
        mode: str = "run_pdf",
    ):
        self.dataset_root = dataset_root
        self.faiss_cases_dir = faiss_cases_dir
        self.faiss_guidelines_dir = faiss_guidelines_dir
        self.top_k = top_k
        self.use_ollama = use_ollama
        self.rf_path = rf_path
        self.debug = debug
        self.mode = mode


def _state_to_dict(st: PipelineState, pdf_name: str) -> Dict[str, Any]:
    """Convert a PipelineState into a clean serialisable dict for the API response."""
    vf = st.validated_features or {}
    risk = st.risk_assessment or {}
    rc = st.retrieved_context or {}
    temporal = st.temporal_analysis or {}
    validation = st.validation or {}

    sim = (rc.get("similar_cases") or {})
    gl = (rc.get("guidelines") or {})

    return {
        "pdf_name": pdf_name,
        "patient_id": vf.get("patient_id"),
        "pod": vf.get("pod"),
        "procedure": vf.get("procedure"),
        "parser_quality": vf.get("extraction_quality"),
        "validation": {
            "status": validation.get("status", "unknown"),
            "errors": validation.get("errors", []),
            "warnings": validation.get("warnings", []),
        },
        "temporal_analysis": {
            "status": temporal.get("status", "unknown"),
            "history_points_used": temporal.get("history_points_used", 0),
            "temperature_trend": temporal.get("temperature_trend"),
            "heart_rate_trend": temporal.get("heart_rate_trend"),
            "wbc_trend": temporal.get("wbc_trend"),
            "spo2_trend": temporal.get("spo2_trend"),
            "expected_ranges_bucket": temporal.get("expected_ranges_bucket"),
            "deviations": temporal.get("deviations") or [],
            "rapid_deterioration": temporal.get("rapid_deterioration"),
            "rapid_deterioration_any": temporal.get("rapid_deterioration_any", False),
        },
        "risk_assessment": {
            "risk_level": risk.get("risk_level", "MEDIUM"),
            "risk_score": risk.get("risk_score", 0),
            "alerts": risk.get("alerts") or [],
            "rule_score": risk.get("rule_score", 0),
            "rule_level": risk.get("rule_level", "MEDIUM"),
            "rule_factors": risk.get("rule_factors") or [],
            "ml_pred": risk.get("ml_pred"),
            "ml_proba": risk.get("ml_proba"),
            "final_factors": risk.get("final_factors") or [],
            "llm_reasoning": risk.get("llm_reasoning"),
        },
        "retrieved_context": {
            "query_used": rc.get("query_used"),
            "similar_cases_citations": sim.get("citations") or [],
            "guidelines_citations": gl.get("citations") or [],
        },
        "final_response": st.final_response or "",
        "errors": st.errors or [],
        "trace": st.trace or [],
    }


# ── Public API ─────────────────────────────────────────────────────────────

_GRAPH = None  # module-level singleton so the graph is compiled only once


def _get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


def run_single_pdf(
    pdf_bytes: bytes,
    pdf_filename: str,
    dataset_root: str = str(DEFAULT_DATASET_ROOT),
    faiss_cases_dir: str = str(DEFAULT_FAISS_CASES_DIR),
    faiss_guidelines_dir: str = str(DEFAULT_FAISS_GUIDELINES_DIR),
    top_k: int = 5,
    use_ollama: bool = False,
    rf_path: str = str(DEFAULT_RF_PATH),
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Accept raw PDF bytes, write to a temp file, run the full pipeline,
    and return a serialisable result dict.
    """
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = Path(tmp.name)

    try:
        args = _Args(
            dataset_root=dataset_root,
            faiss_cases_dir=faiss_cases_dir,
            faiss_guidelines_dir=faiss_guidelines_dir,
            top_k=top_k,
            use_ollama=use_ollama,
            rf_path=rf_path,
            debug=debug,
        )
        graph = _get_graph()
        state = run_one_pdf(graph, tmp_path, args)
        return _state_to_dict(state, pdf_filename)
    finally:
        tmp_path.unlink(missing_ok=True)


def run_pdf_list(
    pdf_items: List[Dict[str, Any]],   # [{"filename": str, "bytes": bytes}, ...]
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Run the pipeline sequentially over a list of PDF dicts.
    Each item must have 'filename' and 'bytes' keys.
    Returns a list of result dicts in the same order.
    """
    results = []
    for item in pdf_items:
        try:
            result = run_single_pdf(
                pdf_bytes=item["bytes"],
                pdf_filename=item["filename"],
                **kwargs,
            )
        except Exception as exc:
            result = {
                "pdf_name": item.get("filename", "unknown"),
                "error": str(exc),
                "risk_assessment": {"risk_level": None},
            }
        results.append(result)
    return results
