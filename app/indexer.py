"""
core/indexer.py — Thin wrappers around build_rag_index.py and build_guidelines_faiss.py.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.build_rag_index import RAGBuilder, project_root_from_cwd   # noqa: E402
from scripts.build_guidelines_faiss import load_docs, main as _guidelines_main  # noqa: E402

from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    from langchain_community.embeddings import HuggingFaceEmbeddings

try:
    from langchain_community.vectorstores import FAISS
except Exception:
    from langchain.vectorstores import FAISS

MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


def rebuild_cases_index(
    dataset_root: str,
    faiss_cases_dir: str,
    chunk_size: int = 600,
    chunk_overlap: int = 80,
) -> Dict[str, Any]:
    """
    Rebuild the patient-cases FAISS index from scratch.
    Returns a summary dict.
    """
    root = Path(dataset_root).resolve()
    builder = RAGBuilder(
        project_root=root.parent if (root.parent / "Data").exists() else root,
        rebuild=True,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    builder.load_pdfs()
    builder.build_records()
    builder.create_vectorstore()

    return {
        "status": "ok",
        "index_type": "cases",
        "chunks_built": len(list(builder.vectordb_dir.glob("*"))),
        "vectordb_dir": str(builder.vectordb_dir),
        "message": f"Cases FAISS index rebuilt successfully at {builder.vectordb_dir}",
    }


def rebuild_guidelines_index(
    guidelines_dir: str,
    faiss_guidelines_dir: str,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> Dict[str, Any]:
    """
    Rebuild the guidelines FAISS index.
    Returns a summary dict.
    """
    g_path = Path(guidelines_dir).resolve()
    out_path = Path(faiss_guidelines_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    docs = load_docs(g_path)
    if not docs:
        return {
            "status": "error",
            "index_type": "guidelines",
            "chunks_built": 0,
            "vectordb_dir": str(out_path),
            "message": f"No guideline documents found in {g_path}",
        }

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(docs)

    emb = HuggingFaceEmbeddings(model_name=MINILM_MODEL)
    store = FAISS.from_documents(chunks, emb)
    store.save_local(str(out_path))

    return {
        "status": "ok",
        "index_type": "guidelines",
        "chunks_built": len(chunks),
        "vectordb_dir": str(out_path),
        "message": f"Guidelines FAISS index rebuilt with {len(chunks)} chunks at {out_path}",
    }


def get_index_status(
    faiss_cases_dir: str,
    faiss_guidelines_dir: str,
) -> Dict[str, Any]:
    cases_path = Path(faiss_cases_dir).resolve()
    guidelines_path = Path(faiss_guidelines_dir).resolve()

    return {
        "cases_index_exists": cases_path.exists() and any(cases_path.iterdir()),
        "cases_index_dir": str(cases_path),
        "guidelines_index_exists": guidelines_path.exists() and any(guidelines_path.iterdir()),
        "guidelines_index_dir": str(guidelines_path),
        "embedding_model": MINILM_MODEL,
        "embedding_dim": EMBEDDING_DIM,
    }
