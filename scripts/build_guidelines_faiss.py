#!/usr/bin/env python3
from pathlib import Path

try:
    from langchain_community.document_loaders import PyPDFLoader, TextLoader
    from langchain_community.vectorstores import FAISS
except Exception:
    from langchain.document_loaders import PyPDFLoader, TextLoader
    from langchain.vectorstores import FAISS

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except Exception:
        from langchain.embeddings import HuggingFaceEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter

MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def load_docs(guidelines_dir: Path):
    docs = []
    for p in guidelines_dir.rglob("*"):
        if p.suffix.lower() == ".pdf":
            dl = PyPDFLoader(str(p)).load()
            for d in dl:
                d.metadata["source_file"] = p.name
            docs.extend(dl)
        elif p.suffix.lower() in [".txt", ".md"]:
            dl = TextLoader(str(p), encoding="utf-8").load()
            for d in dl:
                d.metadata["source_file"] = p.name
            docs.extend(dl)
    return docs

def main():
    guidelines_dir = Path("Data/synthetic_dataset/guidelines").resolve()
    out_dir = Path("Data/rag_artifacts/vectordb_guidelines_minilm_384").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    docs = load_docs(guidelines_dir)
    if not docs:
        raise RuntimeError(f"No guideline docs found under {guidelines_dir}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    emb = HuggingFaceEmbeddings(model_name=MINILM_MODEL)
    store = FAISS.from_documents(chunks, emb)
    store.save_local(str(out_dir))

    print(f"Built guidelines FAISS index with {len(chunks)} chunks")
    print(f"Saved to {out_dir}")

if __name__ == "__main__":
    main()
