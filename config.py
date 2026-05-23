import os
from dotenv import load_dotenv
# Load .env file
load_dotenv()

# ── Gemini API ──────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ── Gemini Model Settings ───────────────────────────────
LLM_MODEL = "gemini-3.5-flash"          # Free & fast
EMBEDDING_MODEL = "models/embedding-001" # Gemini embedding model
TEMPERATURE = 0.5                        # Lower = more factual answers

# ── Paths ────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
VECTORSTORE_DIR = os.path.join(BASE_DIR, "vectorstore")

# ── Text Splitting ──────────────────────────────────────
CHUNK_SIZE = 300       # Characters per chunk
CHUNK_OVERLAP = 20     # Overlap between chunks

# ── Retriever ───────────────────────────────────────────
TOP_K_RESULTS = 5       # How many chunks to retrieve per query

# ── Flask ───────────────────────────────────────────────
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_DEBUG = True

# ── Streamlit ───────────────────────────────────────────
STREAMLIT_TITLE = "RAG Chatbot"
STREAMLIT_ICON = "🤖"

# ── Verify API key loaded ───────────────────────────────
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found. Check your .env file.")