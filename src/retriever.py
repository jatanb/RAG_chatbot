import os
import sys

# ── Fix path ────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from config import TOP_K_RESULTS
from src.embeddings import get_embedding_model
from src.vector_store import load_vectorstore


def get_retriever(vectorstore, top_k=TOP_K_RESULTS):
    """Convert vectorstore into a LangChain retriever"""

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k}
    )


    print(f"Retriever ready — fetching top {top_k} chunks per query")
    return retriever


def retrieve_documents(retriever, query):
    """Retrieve relevant chunks for a given query"""

    print(f"\nSearching for: '{query}'")
    docs = retriever.invoke(query)

    print(f"Retrieved {len(docs)} chunks")
    return docs


def format_retrieved_docs(docs):
    """Format retrieved chunks into readable context string"""

    context = ""
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        context += f"\n[Source {i+1}: {os.path.basename(source)}, Page {page}]\n"
        context += doc.page_content
        context += "\n" + "-" * 40

    return context


# ── Quick test ───────────────────────────────────────────
if __name__ == "__main__":

    # Step 1 - Load embedding model
    embedding_model = get_embedding_model()

    # Step 2 - Load vectorstore
    vectorstore = load_vectorstore(embedding_model)

    # Step 3 - Get retriever
    retriever = get_retriever(vectorstore)

    # Step 4 - Test with sample questions
    test_queries = [
        "How many sick leaves does an employee get?",
        "What is the work from home policy?",
        "What is the notice period for resignation?"
    ]

    for query in test_queries:
        print("\n" + "=" * 50)
        docs = retrieve_documents(retriever, query)
        context = format_retrieved_docs(docs)
        print(context)