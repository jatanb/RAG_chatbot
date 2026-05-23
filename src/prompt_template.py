import os
import sys

# ── Fix path ────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from langchain_core.prompts import PromptTemplate


def get_rag_prompt():
    """Main RAG prompt — answers questions using retrieved context"""

    template = """
You are a friendly HR Policy Assistant chatbot for a company.
Your ONLY job is to answer questions about company HR policies.

STRICT RULES — follow every single one:

1. NEVER mention file names like "hr_policy.pdf" or "document.pdf"
2. NEVER say "according to", "based on", "as per the document/pdf/context"
3. NEVER mention page numbers
4. NEVER use words like "chunks", "context", "provided text"
5. Answer naturally as if YOU personally know the company policies
6. Use bullet points for lists, keep answers under 5 lines
7. Be warm, friendly and professional

OUT OF SCOPE RULE — MOST IMPORTANT:
If the question is NOT related to HR policies, company rules,
leaves, attendance, salary, benefits, resignation, conduct,
or workplace topics — respond with EXACTLY this:
"I'm only able to answer questions about company HR policies.
For other topics, please consult the right person or resource. 😊
Is there anything about our HR policies I can help you with?"

DO NOT attempt to answer out of scope questions at all.
DO NOT search the documents for out of scope questions.

POLICY CONTEXT:
{context}

CHAT HISTORY:
{chat_history}

EMPLOYEE QUESTION:
{question}

YOUR ANSWER:
"""

    prompt = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=template
    )

    return prompt


def get_standalone_question_prompt():
    """
    Converts follow-up questions into standalone questions
    using chat history — important for multi-turn conversations

    Example:
    Chat history: Q: What is sick leave? A: 12 days per year
    Follow-up: "Can I carry it forward?"
    Standalone: "Can sick leave be carried forward to next year?"
    """

    template = """
Given the chat history and a follow-up question below,
rephrase the follow-up question to be a complete standalone question.
Do NOT answer the question — just rephrase it.

CHAT HISTORY:
{chat_history}

FOLLOW-UP QUESTION:
{question}

STANDALONE QUESTION:
"""

    prompt = PromptTemplate(
        input_variables=["chat_history", "question"],
        template=template
    )

    return prompt


def format_prompt(prompt, context, question, chat_history=""):
    """Format prompt with actual values for testing"""

    formatted = prompt.format(
        context=context,
        question=question,
        chat_history=chat_history
    )

    return formatted


# ── Quick test ───────────────────────────────────────────
if __name__ == "__main__":

    # Test RAG prompt
    rag_prompt = get_rag_prompt()

    sample_context = """
    [Source 1: leave_policy.pdf, Page 2]
    Employees are entitled to 12 days sick leave per year.
    Sick leave cannot be carried forward to the next year.
    """

    sample_question = "How many sick leaves do I get?"
    sample_history = "Human: What types of leave exist?\nAI: There are sick, casual and earned leaves."

    formatted = format_prompt(
        rag_prompt,
        context=sample_context,
        question=sample_question,
        chat_history=sample_history
    )

    print("=" * 50)
    print("FORMATTED RAG PROMPT:")
    print("=" * 50)
    print(formatted)

    # Test standalone question prompt
    standalone_prompt = get_standalone_question_prompt()
    print("\n" + "=" * 50)
    print("STANDALONE QUESTION PROMPT:")
    print("=" * 50)
    print(standalone_prompt.template)