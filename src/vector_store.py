import os
import sys

# ── Fix path ────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from config import VECTORSTORE_DIR
from langchain_community.vectorstores import FAISS
import chromadb


def save_vectorstore(chunks, embedding_model):
    """Create FAISS vectorstore from chunks and save to disk"""

    print(f"\nCreating FAISS vectorstore from {len(chunks)} chunks...")

    # Create vectorstore from chunks
    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embedding_model
    )

    # Save to disk
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    vectorstore.save_local(VECTORSTORE_DIR)

    print(f"Vectorstore saved to: {VECTORSTORE_DIR}")
    print(f"Total vectors stored: {vectorstore.index.ntotal}")
    return vectorstore


def load_vectorstore(embedding_model):
    """Load existing FAISS vectorstore from disk"""

    if not os.path.exists(VECTORSTORE_DIR):
        raise FileNotFoundError(
            f"Vectorstore not found at {VECTORSTORE_DIR}. Run ingest.py first!"
        )

    print("Loading vectorstore from disk...")

    vectorstore = FAISS.load_local(
        VECTORSTORE_DIR,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )

    print(f"Vectorstore loaded! Total vectors: {vectorstore.index.ntotal}")
    return vectorstore


def search_vectorstore(vectorstore, query, top_k=5):
    """Search vectorstore for relevant chunks"""

    print(f"\nSearching for: '{query}'")
    results = vectorstore.similarity_search(query, k=top_k)

    print(f"Found {len(results)} relevant chunks")
    return results


# ── Quick test ───────────────────────────────────────────
if __name__ == "__main__":

    # Import required modules
    from document_loader import load_all_documents
    from text_splitter import split_documents
    from embeddings import get_embedding_model

    # Step 1 - Load documents
    documents = load_all_documents()

    # Step 2 - Split into chunks
    chunks = split_documents(documents)

    # Step 3 - Load embedding model
    embedding_model = get_embedding_model()

    # Step 4 - Save vectorstore
    vectorstore = save_vectorstore(chunks, embedding_model)

    # Step 5 - Test search
    results = search_vectorstore(
        vectorstore,
        query="How many sick leaves does an employee get?",
        top_k=3
    )

    print("\n--- Search Results ---")
    for i, doc in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"Source: {doc.metadata}")
        print(f"Content: {doc.page_content[:200]}")