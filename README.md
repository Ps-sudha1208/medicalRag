# medicalRAG 

A LangGraph-based multi-agent clinical risk assessment pipeline using RAG.

## Features
- 6-agent LangGraph orchestration
- Parser with regex → spaCy → Ollama fallbacks
- FAISS + MiniLM retrieval (cases + guidelines)
- Temporal trend & rapid deterioration detection
- Rule-based + ML + optional LLM risk assessment
- LangSmith tracing support
- Full evaluation metrics pipeline

## Project Structure
medicalRAG/
├── Data/
├── scripts/
├── results/


## Run pipeline (example)
```bash
python scripts/clinical_langgraph_pipeline.py \
  --mode run_pdf \
  --pdf Data/synthetic_dataset/pdfs/PT0002_POD17_LOW_20260206.pdf \
  --dataset-root Data/synthetic_dataset \
  --faiss-cases-dir Data/rag_artifacts/vectordb_faiss_minilm_384 \
  --faiss-guidelines-dir Data/rag_artifacts/vectordb_guidelines_minilm_384 \
  --top-k 5 --debug
