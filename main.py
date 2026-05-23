import os
import sys

# ── Fix path ────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import streamlit as st
from src.chat_engine import ChatEngine
from config import STREAMLIT_TITLE, STREAMLIT_ICON


# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title=STREAMLIT_TITLE,
    page_icon=STREAMLIT_ICON,
    layout="centered"
)


# ── CSS ──────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .chat-info {
        background: #f0f2f6;
        padding: 0.8rem;
        border-radius: 8px;
        font-size: 0.85rem;
        color: #555;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Load engine ONCE using cache ─────────────────────────
@st.cache_resource
def load_engine():
    """
    Loads ChatEngine only once.
    Streamlit reruns this file on every interaction
    but cache_resource prevents reloading the heavy model.
    """
    print("Loading ChatEngine — this runs only once!")
    engine = ChatEngine()
    engine.initialize()
    return engine


# ── Get cached engine ────────────────────────────────────
engine = load_engine()


# ── Session state for messages ───────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "exchange_count" not in st.session_state:
    st.session_state.exchange_count = 0


# ── Header ───────────────────────────────────────────────
st.markdown(f"""
<div class="main-header">
    <h1>{STREAMLIT_ICON} {STREAMLIT_TITLE}</h1>
    <p>Ask me anything about company HR policies</p>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Controls")

    # Engine status
    status = engine.get_status()
    if status["is_ready"]:
        st.success("✅ Engine Ready")
    else:
        st.error("❌ Engine Not Ready")
        st.info("Run ingest.py first!")

    st.divider()

    # Exchange counter
    st.metric(
        label="Total Exchanges",
        value=st.session_state.exchange_count
    )

    st.divider()

    # Clear button
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.exchange_count = 0
        engine.clear_memory()
        st.success("Conversation cleared!")
        st.rerun()

    st.divider()

    # Sample questions
    st.subheader("💡 Try asking")
    sample_questions = [
        "How many sick leaves do I get?",
        "What is the WFH policy?",
        "What is the notice period?",
        "How do I apply for maternity leave?",
        "What are the working hours?"
    ]

    for q in sample_questions:
        if st.button(q, use_container_width=True, key=q):
            st.session_state.pending_question = q
            st.rerun()

    st.divider()
    st.caption("Built with LangChain + Gemini + FAISS")




# ── Display existing messages ────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ── Handle sidebar question clicks ──────────────────────
if "pending_question" in st.session_state:
    question = st.session_state.pending_question
    del st.session_state.pending_question

    with st.chat_message("user"):
        st.markdown(question)

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = engine.chat(question)
        st.markdown(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    st.session_state.exchange_count += 1
    st.rerun()


# ── Chat input ───────────────────────────────────────────
if question := st.chat_input("Ask me anything"):

    with st.chat_message("user"):
        st.markdown(question)

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = engine.chat(question)
        st.markdown(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    st.session_state.exchange_count += 1
    st.rerun()