import os
import sys
import re

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
    """

    def __init__(self):
        self.chain = None
        self.memory = None
        self.is_ready = False
        print("ChatEngine created.")

    def initialize(self):
        """Load all components and prepare chatbot"""

        print("\n" + "=" * 50)
        print("  INITIALIZING CHAT ENGINE")
        print("=" * 50)

        try:
            print("\n[1/4] Loading embedding model...")
            embedding_model = get_embedding_model()

            print("\n[2/4] Loading vectorstore...")
            vectorstore = load_vectorstore(embedding_model)

            print("\n[3/4] Building retriever...")
            retriever = get_retriever(vectorstore)

            print("\n[4/4] Building LLM chain...")
            self.chain = build_chain(retriever)

            self.memory = ChatMemory(max_history=10)
            self.is_ready = True

            print("\n" + "=" * 50)
            print("  CHAT ENGINE READY!")
            print("=" * 50)

        except FileNotFoundError:
            print("\nERROR: Vectorstore not found!")
            print("Please run: python ingest.py")
            self.is_ready = False

        except Exception as e:
            print(f"\nERROR initializing chat engine: {e}")
            self.is_ready = False

    # ── Abbreviation expander ────────────────────────────
    def expand_question(self, question):
        """
        Expands short forms so FAISS finds correct chunks.
        Example: "WFH policy" → "work from home policy"
        """

        expansions = {
            "wfh":               "work from home policy",
            "wfh policy":        "work from home policy",
            "work from home":    "work from home policy",
            "pl":                "paid leave earned leave",
            "cl":                "casual leave policy",
            "sl":                "sick leave policy",
            "el":                "earned leave policy",
            "al":                "annual leave policy",
            "lop":               "loss of pay leave without pay",
            "lwp":               "leave without pay",
            "pf":                "provident fund policy",
            "ctc":               "cost to company salary structure",
            "fnf":               "full and final settlement",
            "noc":               "no objection certificate",
            "kra":               "key result areas performance review",
            "kpi":               "key performance indicator",
            "appraisal":         "performance review increment policy",
            "hike":              "salary increment policy",
            "notice":            "notice period resignation policy",
            "resign":            "resignation exit policy",
            "probation":         "probation period policy",
            "dress code":        "dress code conduct policy",
            "reimbursement":     "expense reimbursement policy",
            "overtime":          "overtime compensation policy",
            "insurance":         "health insurance benefits policy",
            "gratuity":          "gratuity policy full final settlement",
        }

        question_lower = question.lower().strip()

        # Check full question match
        if question_lower in expansions:
            expanded = expansions[question_lower]
            print(f"Expanded: '{question}' → '{expanded}'")
            return expanded

        # Check partial match — word by word
        words = question_lower.split()
        for short, full in expansions.items():
            short_words = short.split()
            if all(w in words for w in short_words):
                expanded = question_lower
                for w in short_words:
                    expanded = expanded.replace(w, "")
                expanded = (full + " " + expanded).strip()
                print(f"Expanded: '{question}' → '{expanded}'")
                return expanded

        return question

    # ── HR scope checker ─────────────────────────────────
    def is_hr_related(self, question):
        """Check if question is HR related"""

        hr_keywords = [
            # Leave
            "leave", "sick", "casual", "annual", "maternity",
            "paternity", "holiday", "vacation", "off", "absence",
            "time off", "days off", "pl", "cl", "sl", "el", "al",
            "lwp", "lop",

            # Work location
            "wfh", "work from home", "remote", "hybrid", "office",
            "work location", "home office", "telework", "from home",
            "working from home",

            # Attendance and hours
            "attendance", "hours", "overtime", "shift", "timing",
            "working hours", "late", "punctuality", "check in",
            "login", "logout", "biometric",

            # Salary and benefits
            "salary", "pay", "bonus", "increment", "appraisal",
            "promotion", "performance", "review", "hike", "raise",
            "benefits", "insurance", "pf", "gratuity", "ctc",
            "package", "compensation", "allowance", "reimbursement",
            "expense", "travel", "medical", "health", "fnf",

            # HR processes
            "resign", "resignation", "notice", "notice period",
            "exit", "probation", "joining", "onboarding",
            "training", "policy", "rule", "regulation",
            "conduct", "dress", "code", "grievance", "complaint",
            "harassment", "discipline", "termination", "transfer",
            "noc", "relieving", "experience letter",

            # General HR
            "hr", "human resource", "employee", "employer",
            "company policy", "workplace", "office policy",
            "staff", "colleague", "manager", "department",
            "holiday list", "weekend", "working day",
            "confirmation", "permanent", "contract",
        ]

        question_lower = question.lower()
        for keyword in hr_keywords:
            if keyword in question_lower:
                return True

        return False

    # ── Answer cleaner ───────────────────────────────────
    def clean_answer(self, answer):
        """
        Removes all PDF file names, page numbers and
        document references from Gemini response.
        100% reliable — Python regex always works.
        """

        # Remove file names like HR_policy.pdf or document.pdf
        answer = re.sub(
            r'\b[\w\-]+\.(pdf|docx|txt)\b', '',
            answer, flags=re.IGNORECASE
        )

        # Remove page references like Page 40, page 5, pg 12
        answer = re.sub(
            r'\b(page|pg|p\.)\s*\d+\b', '',
            answer, flags=re.IGNORECASE
        )

        # Remove source references like [Source 1: ...]
        answer = re.sub(
            r'\[Source\s*\d*:.*?\]', '',
            answer, flags=re.IGNORECASE
        )

        # Remove "according to the document/pdf/context"
        answer = re.sub(
            r'according to (the )?(document|pdf|context|provided|given|uploaded)[\w\s]*[,.]?',
            '', answer, flags=re.IGNORECASE
        )

        # Remove "based on the document/pdf/context"
        answer = re.sub(
            r'based on (the )?(document|pdf|context|provided|given|uploaded)[\w\s]*[,.]?',
            '', answer, flags=re.IGNORECASE
        )

        # Remove "as per the document/pdf"
        answer = re.sub(
            r'as per (the )?(document|pdf|context|provided|given)[\w\s]*[,.]?',
            '', answer, flags=re.IGNORECASE
        )

        # Remove "this can be found in..."
        answer = re.sub(
            r'this (information|policy|detail|can)[\w\s]*(found|available|in|from)[\w\s\.,]*[,.]?',
            '', answer, flags=re.IGNORECASE
        )

        # Remove "in the HR_policy document" type phrases
        answer = re.sub(
            r'in (the )?[\w_\-]+\.(pdf|docx|txt)',
            '', answer, flags=re.IGNORECASE
        )

        # Remove leftover commas and dots
        answer = re.sub(r'\s+', ' ', answer)
        answer = re.sub(r',\s*\.', '.', answer)
        answer = re.sub(r'\.\s*\.', '.', answer)
        answer = re.sub(r'^\s*[,\.]\s*', '', answer)
        answer = answer.strip()

        return answer

    # ── Main chat function ───────────────────────────────
    def chat(self, question):
        """
        Main chat function — handles everything:
        1. Expand abbreviations (WFH → work from home)
        2. Check if HR related
        3. Get answer from RAG chain
        4. Clean answer (remove PDF names, pages)
        5. Save to memory
        """

        if not self.is_ready:
            return "Chat engine not ready. Please run ingest.py first."

        if not question or question.strip() == "":
            return "Please ask a valid question."

        # Step 1 — Expand abbreviations
        expanded_question = self.expand_question(question)

        # Step 2 — Check if HR related
        if not self.is_hr_related(expanded_question):
            return (
                "I'm specialized in company HR policies only — "
                "that's a bit outside my area! 😊\n\n"
                "Feel free to ask me anything about leaves, "
                "WFH, salary, resignation or any other HR topic!"
            )

        try:
            # Step 3 — Get answer from RAG chain
            answer = get_answer(
                chain=self.chain,
                question=expanded_question,
                chat_history=self.memory.get_history()
            )

            # Step 4 — Clean answer
            answer = self.clean_answer(answer)

            # Step 5 — Save original question to memory
            self.memory.add_exchange(question, answer)

            return answer

        except Exception as e:
            print(f"Error getting answer: {e}")
            return "Sorry, something went wrong. Please try again."

    # ── Chat with history from Streamlit ─────────────────
    def chat_with_history(self, question, history):
        """
        Chat using history passed from Streamlit session state.
        Keeps Streamlit UI and engine memory in sync.
        """

        if not self.is_ready:
            return "Chat engine not ready. Please run ingest.py first."

        if not question or question.strip() == "":
            return "Please ask a valid question."

        # Step 1 — Expand abbreviations
        expanded_question = self.expand_question(question)

        # Step 2 — Check scope
        if not self.is_hr_related(expanded_question):
            return (
                "I'm specialized in company HR policies only — "
                "that's a bit outside my area! 😊\n\n"
                "Feel free to ask me anything about leaves, "
                "WFH, salary, resignation or any other HR topic!"
            )

        try:
            # Convert Streamlit message format to tuple format
            # Streamlit: [{"role": "user", "content": "..."}]
            # LangChain: [("question", "answer"), ...]
            chat_history = []
            messages = history.copy()

            for i in range(0, len(messages) - 1, 2):
                if i + 1 < len(messages):
                    human = messages[i]["content"]
                    ai = messages[i + 1]["content"]
                    chat_history.append((human, ai))

            # Get answer
            answer = get_answer(
                chain=self.chain,
                question=expanded_question,
                chat_history=chat_history
            )

            # Clean answer
            answer = self.clean_answer(answer)

            return answer

        except Exception as e:
            print(f"Error: {e}")
            return "Sorry, something went wrong. Please try again."

    # ── Memory controls ──────────────────────────────────
    def clear_memory(self):
        """Clear conversation history"""
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
        """Check engine status"""
        return {
            "is_ready": self.is_ready,
            "exchanges": self.memory.get_exchange_count() if self.memory else 0
        }


# ── Quick test ───────────────────────────────────────────
if __name__ == "__main__":

    engine = ChatEngine()
    engine.initialize()

    if engine.is_ready:

        test_questions = [
            "WFH policy",
            "wfh",
            "How many sl do I get?",
            "what is cl?",
            "What is notice period?",
            "how to get 90lpa",
            "who is virat kohli",
        ]

        print("\n" + "=" * 50)
        print("TESTING CHAT ENGINE")
        print("=" * 50)

        for q in test_questions:
            print(f"\nYou: {q}")
            answer = engine.chat(q)
            print(f"Bot: {answer}")
            print("-" * 40)