"""
Build RAG Index + Extraction/Validation from Synthetic Post-Op PDFs
(FAISS + MiniLM ONLY, cleaned structure)

Inputs:
  Data/synthetic_dataset/
    pdfs/
    jsons/

Outputs:
  Data/rag_artifacts/
    extraction_logs/
    vectordb_faiss_minilm_384/

Dependencies:
  pip install langchain langchain-community langchain-text-splitters sentence-transformers pypdf numpy faiss-cpu
Optional:
  pip install langchain-huggingface
"""

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

# --- LangChain imports ---
try:
    from langchain_community.document_loaders import PyPDFLoader
except Exception:
    from langchain.document_loaders import PyPDFLoader

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

# Prefer langchain-huggingface if installed
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


# -----------------------------
# Config (FAISS + MiniLM)
# -----------------------------
EMBEDDING_PRESET = "minilm"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 384 dim
EXPECTED_DIM = 384

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
    raise FileNotFoundError("Could not locate project root with a Data/ folder.")


class RAGBuilder:
    def __init__(self, project_root: Path, rebuild: bool = False, chunk_size: int = 600, chunk_overlap: int = 80):
        self.root = project_root
        self.data_dir = self.root / "Data"
        self.dataset_root = self.data_dir / "synthetic_dataset"
        self.pdf_dir = self.dataset_root / "pdfs"
        self.json_dir = self.dataset_root / "jsons"

        self.rag_root = self.data_dir / "rag_artifacts"
        self.logs_dir = self.rag_root / "extraction_logs"
        self.vectordb_dir = self.rag_root / f"vectordb_faiss_{EMBEDDING_PRESET}_{EXPECTED_DIM}"

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.rebuild = rebuild

        self.documents = []
        self.records: List[Dict[str, Any]] = []
        self.vectorstore = None

        self._ensure_folders()
        if self.rebuild and self.vectordb_dir.exists():
            shutil.rmtree(self.vectordb_dir)
        self.vectordb_dir.mkdir(parents=True, exist_ok=True)

        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self._assert_embedding_dim()

    def _ensure_folders(self):
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.json_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.rag_root.mkdir(parents=True, exist_ok=True)

    def _assert_embedding_dim(self):
        dim = len(self.embeddings.embed_query("dim probe"))
        if dim != EXPECTED_DIM:
            raise ValueError(f"MiniLM expected dim={EXPECTED_DIM}, but got dim={dim}. Check model / environment.")

    def load_pdfs(self) -> int:
        pdf_files = sorted(self.pdf_dir.glob("*.pdf"))
        if not pdf_files:
            raise FileNotFoundError(f"No PDFs found in: {self.pdf_dir}")

        print(f"Loading PDFs from: {self.pdf_dir} ({len(pdf_files)} files)")
        self.documents = []

        for pdf_path in pdf_files:
            loader = PyPDFLoader(str(pdf_path))
            docs = loader.load()
            for d in docs:
                d.metadata["source_file"] = pdf_path.name
                d.metadata["source_stem"] = pdf_path.stem
                d.metadata["dataset_root"] = str(self.dataset_root)
            self.documents.extend(docs)

        print(f"✓ Loaded {len(self.documents)} pages total")
        return len(self.documents)

    def _safe_float(self, x: str) -> Optional[float]:
        try:
            return float(x)
        except Exception:
            return None

    def _safe_int(self, x: str) -> Optional[int]:
        try:
            return int(x)
        except Exception:
            return None

    def extract_features(self, text: str) -> Dict[str, Any]:
        t = re.sub(r"[ \t]+", " ", text)
        t = re.sub(r"\n+", "\n", t)

        def grab(pattern: str) -> Optional[str]:
            m = re.search(pattern, t, flags=re.IGNORECASE | re.DOTALL)
            return m.group(1).strip() if m else None

        features: Dict[str, Any] = {}
        features["patient_id"] = grab(r"Patient ID:\s*([A-Z]{2}\d{4})")
        pod_str = grab(r"Post-Op Day\s*(\d+)")
        features["pod"] = self._safe_int(pod_str) if pod_str else None
        features["procedure"] = grab(r"Procedure:\s*([^\n]+)")
        features["surgeon"] = grab(r"Surgeon:\s*([^\n]+)")
        features["timestamp"] = grab(r"Timestamp:\s*([^\n]+)")

        temp = grab(r"Temperature\s*([0-9]+(?:\.[0-9]+)?)")
        hr = grab(r"Heart Rate\s*(\d+)")
        bp = re.search(r"Blood Pressure\s*(\d{2,3})/(\d{2,3})", t, flags=re.IGNORECASE)
        rr = grab(r"Respiratory Rate\s*(\d+)")
        spo2 = grab(r"SpO2\s*(\d+)")
        pain = grab(r"Pain Score\s*(\d+)")

        features["temperature"] = self._safe_float(temp) if temp else None
        features["heart_rate"] = self._safe_int(hr) if hr else None
        if bp:
            features["bp_systolic"] = self._safe_int(bp.group(1))
            features["bp_diastolic"] = self._safe_int(bp.group(2))
        else:
            features["bp_systolic"] = None
            features["bp_diastolic"] = None

        features["respiratory_rate"] = self._safe_int(rr) if rr else None
        features["spo2"] = self._safe_int(spo2) if spo2 else None
        features["pain_score"] = self._safe_int(pain) if pain else None

        wbc = grab(r"\bWBC\s*([0-9]+(?:\.[0-9]+)?)")
        hgb = grab(r"Hemoglobin\s*([0-9]+(?:\.[0-9]+)?)")
        features["wbc"] = self._safe_float(wbc) if wbc else None
        features["hemoglobin"] = self._safe_float(hgb) if hgb else None

        notes = grab(r"Clinical Notes\s*(.+?)Automated Risk")
        if not notes:
            notes = grab(r"Clinical Notes\s*(.+)$")
        features["clinical_notes"] = notes

        return features

    def validate_features(self, features: Dict[str, Any]) -> Tuple[List[str], List[str], float]:
        required = ["patient_id", "pod", "procedure", "temperature", "heart_rate", "spo2", "wbc", "pain_score"]
        present = sum(1 for k in required if features.get(k) is not None)

        errors: List[str] = []
        warnings: List[str] = []
        for k in required:
            if features.get(k) is None:
                warnings.append(f"missing_required:{k}")

        for k, (lo, hi) in VALIDATION_RULES.items():
            v = features.get(k)
            if v is None:
                continue
            if not (lo <= float(v) <= hi):
                errors.append(f"out_of_range:{k}={v} not in [{lo},{hi}]")

        completeness = present / float(len(required))
        return errors, warnings, completeness

    def load_ground_truth(self, source_stem: str) -> Optional[Dict[str, Any]]:
        jp = self.json_dir / f"{source_stem}.json"
        if not jp.exists():
            return None
        try:
            return json.loads(jp.read_text(encoding="utf-8"))
        except Exception:
            return None

    def build_records(self) -> int:
        if not self.documents:
            raise ValueError("No documents loaded. Run load_pdfs() first.")

        self.records = []
        for d in self.documents:
            stem = d.metadata.get("source_stem")
            extracted = self.extract_features(d.page_content)
            errors, warnings, completeness = self.validate_features(extracted)
            truth = self.load_ground_truth(stem) if stem else None

            self.records.append({
                "source_file": d.metadata.get("source_file"),
                "source_stem": stem,
                "page": d.metadata.get("page"),
                "extracted_features": extracted,
                "validation_errors": errors,
                "validation_warnings": warnings,
                "completeness_score": completeness,
                "has_ground_truth": truth is not None,
                "ground_truth": truth,
            })

        print(f"✓ Built {len(self.records)} page-level records")
        return len(self.records)

    def evaluate_extraction(self) -> Dict[str, Any]:
        fields = ["temperature", "heart_rate", "spo2", "wbc", "hemoglobin", "pain_score", "pod"]
        abs_errors = {f: [] for f in fields}
        matched = 0
        total_with_truth = 0

        for r in self.records:
            gt = r.get("ground_truth")
            if not gt:
                continue
            total_with_truth += 1
            ex = r["extracted_features"]
            dob = gt.get("daily_observation", {})

            for f in fields:
                exv = ex.get(f)
                gtv = dob.get(f)
                if exv is None or gtv is None:
                    continue
                abs_errors[f].append(abs(float(exv) - float(gtv)))

            if ex.get("patient_id") == gt.get("patient_profile", {}).get("patient_id") and ex.get("pod") == dob.get("pod"):
                matched += 1

        report = {
            "dataset_root": str(self.dataset_root),
            "pdf_dir": str(self.pdf_dir),
            "json_dir": str(self.json_dir),
            "rag_artifacts_root": str(self.rag_root),
            "vectordb_backend": "faiss",
            "embedding_preset": EMBEDDING_PRESET,
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dim": EXPECTED_DIM,
            "vectordb_dir": str(self.vectordb_dir),
            "pdf_count": len(list(self.pdf_dir.glob("*.pdf"))),
            "json_count": len(list(self.json_dir.glob("*.json"))),
            "total_records": len(self.records),
            "records_with_ground_truth": total_with_truth,
            "patient_id_and_pod_exact_match_count": matched,
            "field_mae": {f: (float(np.mean(v)) if v else None) for f, v in abs_errors.items()},
            "validation_error_rate": float(np.mean([1 if r["validation_errors"] else 0 for r in self.records])) if self.records else 0.0,
            "avg_completeness_score": float(np.mean([r["completeness_score"] for r in self.records])) if self.records else 0.0,
        }
        return report

    def write_logs(self, report: Dict[str, Any]) -> None:
        (self.logs_dir / "evaluation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

        out_jsonl = self.logs_dir / "extracted_records.jsonl"
        with out_jsonl.open("w", encoding="utf-8") as f:
            for r in self.records:
                row = dict(r)
                if not row.get("has_ground_truth"):
                    row.pop("ground_truth", None)
                f.write(json.dumps(row) + "\n")

        print(f"✓ Wrote evaluation report: {(self.logs_dir / 'evaluation_report.json').resolve()}")
        print(f"✓ Wrote extracted records:  {out_jsonl.resolve()}")

    def create_vectorstore(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        chunks = splitter.split_documents(self.documents)
        print(f"Split into {len(chunks)} chunks (chunk_size={self.chunk_size}, overlap={self.chunk_overlap})")
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        self.vectorstore.save_local(str(self.vectordb_dir))
        print(f"✓ FAISS index saved at: {self.vectordb_dir.resolve()}")

    def sample_retrieval(self, query: str, k: int = 3):
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        if hasattr(retriever, "invoke"):
            docs = retriever.invoke(query)
        else:
            docs = retriever.get_relevant_documents(query)

        return {
            "query": query,
            "k": k,
            "sources": [d.metadata.get("source_file") for d in docs],
            "snippet": (docs[0].page_content[:200].replace("\n", " ") + "...") if docs else "No results",
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="Delete FAISS folder and rebuild fresh.")
    parser.add_argument("--chunk-size", type=int, default=600)
    parser.add_argument("--chunk-overlap", type=int, default=80)
    args = parser.parse_args()

    root = project_root_from_cwd()
    prep = RAGBuilder(root, rebuild=args.rebuild, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)

    print("\nSTEP 0) Paths")
    print(f"Project root:       {root}")
    print(f"Dataset root:       {prep.dataset_root}")
    print(f"PDFs:              {prep.pdf_dir}")
    print(f"JSONs:             {prep.json_dir}")
    print(f"Artifacts:         {prep.rag_root}")
    print(f"Vector DB folder:  {prep.vectordb_dir}")

    print("\nSTEP 1) Load PDFs")
    prep.load_pdfs()

    print("\nSTEP 2) Extract + Validate + Join JSON")
    prep.build_records()

    print("\nSTEP 3) Evaluate")
    report = prep.evaluate_extraction()
    print(json.dumps(report, indent=2))

    print("\nSTEP 3.1) Write logs")
    prep.write_logs(report)

    print("\nSTEP 4) Build FAISS")
    prep.create_vectorstore()

    print("\nSTEP 5) Sample queries")
    for q in [
        "Which patients have RED FLAG sepsis?",
        "Show notes with wound dehiscence",
        "Find cases with fever and high WBC",
        "SpO2 below 90",
    ]:
        res = prep.sample_retrieval(q, k=3)
        print("\n---")
        print(f"Query:   {res['query']}")
        print(f"Sources: {', '.join([s for s in res['sources'] if s])}")
        print(f"Snippet: {res['snippet']}")


if __name__ == "__main__":
    main()
