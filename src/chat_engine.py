import os
import sys

# ── Fix path ────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.embeddings import get_embedding_model
from src.vector_store import load_vectorstore
from src.retriever import get_retriever
from src.llm_chain import build_chain, get_answer
from src.memory import ChatMemory


class ChatEngine:
    """
    Main RAG ChatEngine
    Connects: VectorStore → Retriever → LLM Chain → Memory
    This is the single class both Flask and Streamlit will use
    """

    def __init__(self):
        self.chain = None
        self.memory = None
        self.is_ready = False
        print("ChatEngine created.")

    def initialize(self):
        """
        Load all components and prepare chatbot
        Call this once before chatting
        """

        print("\n" + "=" * 50)
        print("  INITIALIZING CHAT ENGINE")
        print("=" * 50)

        try:
            # Step 1 - Load embedding model
            print("\n[1/4] Loading embedding model...")
            embedding_model = get_embedding_model()

            # Step 2 - Load vectorstore
            print("\n[2/4] Loading vectorstore...")
            vectorstore = load_vectorstore(embedding_model)

            # Step 3 - Build retriever
            print("\n[3/4] Building retriever...")
            retriever = get_retriever(vectorstore)

            # Step 4 - Build LLM chain
            print("\n[4/4] Building LLM chain...")
            self.chain = build_chain(retriever)

            # Step 5 - Initialize memory
            self.memory = ChatMemory(max_history=100)

            self.is_ready = True

            print("\n" + "=" * 50)
            print("  CHAT ENGINE READY!")
            print("=" * 50)

        except FileNotFoundError:
            print("\nERROR: Vectorstore not found!")
            print("Please run ingest.py first:")
            print("  python ingest.py")
            self.is_ready = False

        except Exception as e:
            print(f"\nERROR initializing chat engine: {e}")
            self.is_ready = False

    def chat(self, question):
        """
        Main chat function
        Takes question → returns answer

        Args:
            question: string from user

        Returns:
            answer: string response from Gemini
        """

        # Check engine is ready
        if not self.is_ready:
            return "Chat engine not ready. Please run ingest.py first."

        # Check question is not empty
        if not question or question.strip() == "":
            return "Please ask a valid question."

        try:
            # Get answer from chain with memory
            answer = get_answer(
                chain=self.chain,
                question=question,
                chat_history=self.memory.get_history()
            )

            # Save to memory
            self.memory.add_exchange(question, answer)

            return answer

        except Exception as e:
            error_msg = f"Error getting answer: {e}"
            print(error_msg)
            return "Sorry, something went wrong. Please try again."

    def clear_memory(self):
        """Clear conversation history — start fresh"""
        if self.memory:
            self.memory.clear()
            return "Conversation cleared! Ask me anything."
        return "Memory not initialized."

    def get_history(self):
        """Get full chat history"""
        if self.memory:
            return self.memory.get_history()
        return []

    def get_status(self):
        """Check if engine is ready"""
        return {
            "is_ready": self.is_ready,
            "exchanges": self.memory.get_exchange_count() if self.memory else 0
        }


# ── Quick test ───────────────────────────────────────────
if __name__ == "__main__":

    # Create and initialize engine
    engine = ChatEngine()
    engine.initialize()

    if engine.is_ready:

        # Simulate full conversation
        questions = [
            "How many sick leaves does an employee get?",
            "Can I carry them forward to next year?",
            "What is the work from home policy?",
            "How many days notice do I need to give before resigning?"
        ]

        print("\n" + "=" * 50)
        print("SIMULATING CONVERSATION")
        print("=" * 50)

        for question in questions:
            print(f"\nYou: {question}")
            answer = engine.chat(question)
            print(f"Bot: {answer}")
            print("-" * 40)

        # Show memory
        engine.memory.show_history()

        # Test clear memory
        print(engine.clear_memory())
        print(f"History after clear: {engine.get_history()}")