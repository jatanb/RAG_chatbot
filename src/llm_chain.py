import os
import sys

# ── Fix path ────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from config import GOOGLE_API_KEY, LLM_MODEL, TEMPERATURE
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda


def get_llm():
    """Initialize Gemini LLM"""

    print(f"Loading Gemini LLM: {LLM_MODEL}")

    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=TEMPERATURE,
        convert_system_message_to_human=True
    )

    print("Gemini LLM loaded successfully!")
    return llm


def format_docs(docs):
    """Format retrieved docs into single context string"""

    context = ""
    for i, doc in enumerate(docs):
        source = os.path.basename(doc.metadata.get("source", "Unknown"))
        page = doc.metadata.get("page", "?")
        context += f"\n[Source {i+1}: {source}, Page {page}]\n"
        context += doc.page_content
        context += "\n" + "-" * 40
    return context


def format_history(chat_history):
    """Format chat history list into string"""

    if not chat_history:
        return "No previous conversation."

    history_str = ""
    for human, ai in chat_history:
        history_str += f"Human: {human}\nAI: {ai}\n\n"
    return history_str.strip()


def build_chain(retriever):
    """Build RAG chain using LCEL for LangChain 1.3.1"""

    print("Building conversation chain...")

    llm = get_llm()

    template = """
You are a helpful HR Assistant for a company.
Answer employee questions based ONLY on the company
policy documents provided below.

RULES:
- Answer ONLY from the context provided
- If answer is not in context say: "I could not find
  this information in company policy. Please contact HR."
- Be clear, friendly and professional
- Mention which document the answer comes from

CONTEXT:
{context}

CHAT HISTORY:
{chat_history}

QUESTION:
{question}

ANSWER:
"""

    prompt = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=template
    )

    # Extract question from input dict
    get_question = RunnableLambda(lambda x: x["question"])

    # Extract and format history from input dict
    get_history = RunnableLambda(lambda x: format_history(x["chat_history"]))

    # Retrieve and format docs based on question
    get_context = RunnableLambda(lambda x: format_docs(retriever.invoke(x["question"])))

    # Final LCEL chain
    chain = (
        {
            "context": get_context,
            "question": get_question,
            "chat_history": get_history
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    print("Conversation chain ready!")
    return chain


def get_answer(chain, question, chat_history=[]):
    """Get answer from RAG chain"""

    print(f"\nQuestion: {question}")

    response = chain.invoke({
        "question": question,
        "chat_history": chat_history
    })

    print(f"\nAnswer: {response}")
    return response


# ── Quick test ───────────────────────────────────────────
if __name__ == "__main__":

    from src.embeddings import get_embedding_model
    from src.vector_store import load_vectorstore
    from src.retriever import get_retriever

    # Step 1 - Load everything
    embedding_model = get_embedding_model()
    vectorstore = load_vectorstore(embedding_model)
    retriever = get_retriever(vectorstore)

    # Step 2 - Build chain
    chain = build_chain(retriever)

    # Step 3 - Test single question
    print("\n" + "=" * 50)
    print("TEST 1 — Single question")
    print("=" * 50)
    answer = get_answer(
        chain,
        question="How many sick leaves does an employee get?",
        chat_history=[]
    )

    # Step 4 - Test follow up question
    print("\n" + "=" * 50)
    print("TEST 2 — Follow up question")
    print("=" * 50)
    chat_history = [("How many sick leaves does an employee get?", answer)]
    get_answer(
        chain,
        question="Can I carry them forward to next year?",
        chat_history=chat_history
    )