import os
import sys

# ── Fix path ────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

from src.document_loader import load_all_documents
from src.text_splitter import split_documents
from src.embeddings import get_embedding_model
from src.vector_store import save_vectorstore


def run_ingestion():
    """
    Full ingestion pipeline:
    PDFs → Load → Split → Embed → Save to FAISS
    Run this ONCE whenever you add new documents
    """

    print("=" * 50)
    print("   RAG INGESTION PIPELINE STARTED")
    print("=" * 50)

    # ── Step 1: Load all documents ───────────────────────
    print("\n[1/4] Loading documents...")
    documents = load_all_documents()
    print(f"✓ Loaded {len(documents)} pages")

    # ── Step 2: Split into chunks ────────────────────────
    print("\n[2/4] Splitting into chunks...")
    chunks = split_documents(documents)
    print(f"✓ Created {len(chunks)} chunks")

    # ── Step 3: Load embedding model ─────────────────────
    print("\n[3/4] Loading embedding model...")
    embedding_model = get_embedding_model()
    print(f"✓ Embedding model ready")

    # ── Step 4: Save to vectorstore ──────────────────────
    print("\n[4/4] Creating and saving vectorstore...")
    vectorstore = save_vectorstore(chunks, embedding_model)
    print(f"✓ Vectorstore saved")

    print("\n" + "=" * 50)
    print("   INGESTION COMPLETE!")
    print("=" * 50)
    print("\nYour RAG pipeline is ready.")
    print("You can now run the chatbot.")
    print(f"Total chunks indexed: {len(chunks)}")

    return vectorstore


# ── Run ──────────────────────────────────────────────────
if __name__ == "__main__":
    run_ingestion()