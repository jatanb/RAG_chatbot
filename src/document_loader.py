import os
import sys
from pathlib import Path


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from config import DATA_DIR

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    WebBaseLoader
)


def load_single_document(file_path: str):

    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    else:
        print(f"  Skipping unsupported file: {file_path}")
        return []

    return loader.load()


def load_all_documents():

    documents = []
    supported = [".pdf", ".txt", ".docx"]

    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Folder not found: {DATA_DIR}")

    files = list(Path(DATA_DIR).iterdir())

    if len(files) == 0:
        raise ValueError("No files found in data/raw/ folder.")

    print(f"Found {len(files)} file(s) in data/raw/")

    for file_path in files:
        if file_path.suffix.lower() in supported:
            print(f"  Loading: {file_path.name}")
            docs = load_single_document(str(file_path))
            documents.extend(docs)
            print(f"  Pages loaded: {len(docs)}")
        else:
            print(f"  Skipping: {file_path.name}")

    print(f"\nTotal pages loaded: {len(documents)}")
    return documents



if __name__ == "__main__":
    docs = load_all_documents()
    print("\n--- Sample from first document ---")
    print(f"Source: {docs[0].metadata}")
    print(f"Content preview:\n{docs[0].page_content[:3]}")