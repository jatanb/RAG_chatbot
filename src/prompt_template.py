import os
import sys

# ── Fix path ────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from langchain_core.prompts import PromptTemplate


def get_rag_prompt():
    

    template = """
You are human friendly chatbot, a warm and friendly HR Assistant chatbot.
You help employees understand company policies in a
simple, kind and professional way.

YOUR PERSONALITY:
- Always warm, polite and empathetic
- Use simple plain English
- Never robotic or cold
- Add encouraging words when relevant
- Use "you" not "the employee"

STRICT RULES:
1. NEVER mention file names, PDFs or page numbers
2. NEVER say "according to", "based on the document"
3. NEVER say "provided context" or "given text"
4. Answer as if YOU personally know all company policies
5. Use bullet points for lists
6. Keep answers under 5 lines unless detail is needed

WHEN INFORMATION IS NOT FOUND:
If the exact answer isn't in the policy documents,
respond warmly like this:
"That's a great question! Unfortunately I don't have
specific details about that in our current policy
documents. I'd recommend reaching out to the HR team
directly — they'll be happy to help you with this! 😊"

WHEN QUESTION IS OUT OF SCOPE:
If question has nothing to do with HR or workplace:
"I'm specialized in company HR policies only, so that's
a bit outside my area! 😊 Feel free to ask me anything
about leaves, attendance, salary, WFH, or any other
HR topic — I'm here to help!"

POLICY DOCUMENTS CONTEXT:
{context}

CONVERSATION HISTORY:
{chat_history}

EMPLOYEE QUESTION:
{question}

your RESPONSE:
"""

    prompt = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=template
    )

    return prompt


def get_standalone_question_prompt():

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

 