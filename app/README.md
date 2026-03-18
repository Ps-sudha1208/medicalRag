# Clinical Risk Pipeline — FastAPI Integration

This document explains how the FastAPI layer sits on top of the existing
LangGraph pipeline and how to run, configure, and use every endpoint.

---

## Project Structure

```
your-repo/
│
├── scripts/                          ← YOUR EXISTING CODE (unchanged)
│   ├── clinical_langgraph_pipeline.py
│   ├── build_rag_index.py
│   ├── build_guidelines_faiss.py
│   ├── evaluate_pipeline.py
│   └── generate_synthetic_dataset.py
│
├── app/                              ← NEW FastAPI layer
│   ├── main.py                       ← FastAPI app + router registration
│   ├── config.py                     ← Centralised settings (.env support)
│   ├── routers/
│   │   ├── analyze.py                ← POST /analyze, /analyze/batch, /jobs/...
│   │   ├── index.py                  ← POST /index/cases/rebuild, /guidelines/rebuild
│   │   ├── model.py                  ← POST /model/train, GET /model/status
│   │   ├── evaluate.py               ← POST /evaluate, GET /evaluate/latest
│   │   └── patients.py               ← GET /patients, /history, /pod
│   ├── core/
│   │   ├── pipeline.py               ← Wraps clinical_langgraph_pipeline.py
│   │   ├── indexer.py                ← Wraps build_rag_index + build_guidelines_faiss
│   │   ├── evaluator.py              ← Wraps evaluate_pipeline.py
│   │   ├── trainer.py                ← Wraps train_random_forest()
│   │   └── patients.py               ← Reads JSON sidecars directly
│   └── schemas/
│       └── models.py                 ← All Pydantic request/response models
│
├── Data/                             ← YOUR EXISTING DATA (unchanged)
│   └── synthetic_dataset/
│       ├── pdfs/
│       ├── jsons/
│       └── guidelines/
│
├── .env                              ← Optional: override default paths
└── requirements.txt
```

---

## Installation

```bash
pip install -r app/requirements.txt
```

---

## Configuration

All paths are configurable via environment variables or a `.env` file in the repo root.
Defaults match your existing directory layout.

```env
# .env (optional — defaults work if you run from repo root)
DATASET_ROOT=Data/synthetic_dataset
FAISS_CASES_DIR=Data/rag_artifacts/vectordb_faiss_minilm_384
FAISS_GUIDELINES_DIR=Data/rag_artifacts/vectordb_guidelines_minilm_384
GUIDELINES_DIR=Data/synthetic_dataset/guidelines
RF_PATH=Data/rag_artifacts/models/risk_rf.joblib
DEFAULT_RESULTS_JSONL=results/results.jsonl
```

---

## Running the Server

```bash
# from repo root
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Interactive Swagger UI: **http://localhost:8000/docs**
OpenAPI JSON schema: **http://localhost:8000/openapi.json**

---

## API Endpoints

### Health

| Method | Path      | Description          |
|--------|-----------|----------------------|
| GET    | `/`       | Liveness check       |
| GET    | `/health` | Health check         |

---

### Analyze

#### `POST /analyze` — Single PDF

Upload one post-op PDF and get a full risk assessment back synchronously.

```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@Data/synthetic_dataset/pdfs/PT0002_POD17_LOW_20260206.pdf" \
  -F "top_k=5" \
  -F "use_ollama=false" \
  -F "debug=false"
```

**Response fields:**
- `risk_assessment.risk_level` — LOW / MEDIUM / HIGH / CRITICAL
- `risk_assessment.alerts` — list of urgent alert strings
- `risk_assessment.ml_pred` — Random Forest prediction (if model exists)
- `temporal_analysis` — trend directions, expected-range deviations, rapid deterioration flags
- `retrieved_context` — citations from similar cases and guidelines
- `final_response` — full human-readable clinical assessment text
- `parser_quality` — 0.0–1.0, fraction of required fields successfully extracted

---

#### `POST /analyze/batch` — Multiple PDFs (async)

Upload several PDFs at once. Returns a `job_id` immediately; processing
happens in the background.

```bash
curl -X POST http://localhost:8000/analyze/batch \
  -F "files=@PT0001_POD1_LOW.pdf" \
  -F "files=@PT0002_POD17_LOW.pdf" \
  -F "top_k=5"
```

**Response:**
```json
{ "job_id": "abc-123", "status": "queued", "total_files": 2 }
```

---

#### `GET /analyze/jobs/{job_id}/status` — Poll progress

```bash
curl http://localhost:8000/analyze/jobs/abc-123/status
```

**Response:**
```json
{ "job_id": "abc-123", "status": "running", "processed": 1, "total_files": 2 }
```

---

#### `GET /analyze/jobs/{job_id}/results` — Retrieve results

Only available once `status == "done"`.

```bash
curl http://localhost:8000/analyze/jobs/abc-123/results
```

---

### Index Management

#### `GET /index/status`

```bash
curl http://localhost:8000/index/status
```

Returns whether each FAISS index exists, its directory, and the embedding model used.

---

#### `POST /index/cases/rebuild`

Triggers a background rebuild of the patient-cases FAISS index from the PDFs.
Safe to call when new PDFs are added to the dataset.

```bash
curl -X POST http://localhost:8000/index/cases/rebuild
```

---

#### `POST /index/guidelines/rebuild`

Triggers a background rebuild of the clinical guidelines FAISS index.

```bash
curl -X POST http://localhost:8000/index/guidelines/rebuild
```

---

### ML Model

#### `GET /model/status`

```bash
curl http://localhost:8000/model/status
```

#### `POST /model/train`

Trains the Random Forest classifier on JSON ground truth sidecars.
Call this after generating a new dataset or adding more patients.

```bash
curl -X POST http://localhost:8000/model/train
```

---

### Evaluation

#### `POST /evaluate`

Run evaluation against a `results.jsonl` file and ground truth sidecars.

```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{"results_jsonl_path": "results/results.jsonl", "dataset_root": "Data/synthetic_dataset"}'
```

**Response includes:**
- `accuracy`, `critical_recall`, `alert_rate`
- `per_tier` — precision / recall / F1 / support for each risk tier
- `confusion_matrix` — rows = true label, cols = predicted label
- `alert_rate_by_tier` — how often alerts fire per predicted tier

---

#### `GET /evaluate/latest`

Returns the most recently cached evaluation result without re-running.

---

### Patients

#### `GET /patients`

Returns a sorted list of all patient IDs in the dataset.

```bash
curl http://localhost:8000/patients
# ["PT0001", "PT0002", "PT0003", ...]
```

---

#### `GET /patients/{patient_id}/history`

Full longitudinal history for one patient, sorted by POD.

```bash
curl http://localhost:8000/patients/PT0002/history
```

---

#### `GET /patients/{patient_id}/pod/{pod}`

Single POD record with full ground truth and patient profile.

```bash
curl http://localhost:8000/patients/PT0002/pod/17
```

---

## Typical Workflow

```
1. Generate dataset          python scripts/generate_synthetic_dataset.py
2. Build FAISS indexes       POST /index/cases/rebuild
                             POST /index/guidelines/rebuild
3. Train ML model            POST /model/train
4. Analyze a PDF             POST /analyze
5. Analyze a full folder     POST /analyze/batch  →  GET /jobs/{id}/results
6. Evaluate results          POST /evaluate
7. Browse patient data       GET /patients/{id}/history
```

---

## Notes

- The LangGraph pipeline graph is compiled **once** at first request and reused
  across all subsequent calls (module-level singleton in `core/pipeline.py`).
- The batch job store is **in-memory**. Jobs are lost on server restart.
  For production, replace `_jobs` dict in `routers/analyze.py` with Redis or a database.
- Ollama is entirely optional. If `use_ollama=false` (default), no external
  services are required beyond the FAISS indexes.
