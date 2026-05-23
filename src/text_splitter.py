import os
import sys
from pathlib import Path

# ── Fix path ────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from config import CHUNK_SIZE, CHUNK_OVERLAP
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = splitter.split_documents(documents)

    print(f"Total chunks created: {len(chunks)}")
    return chunks





# ── Quick test ──────────────────────────────────────────
if __name__ == "__main__":

    # Import document loader to test together
    from document_loader import load_all_documents

    # Step 1 - Load all PDFs
    documents = load_all_documents()

    # Step 2 - Split into chunks
    chunks = split_documents(documents)