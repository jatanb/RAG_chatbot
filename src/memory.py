import os
import sys

# ── Fix path ────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)


class ChatMemory:
    """
    Manages conversation history for the chatbot
    Stores (question, answer) pairs for multi-turn conversations
    """

    def __init__(self, max_history=10):
        """
        max_history = how many exchanges to remember
        After limit, oldest messages are removed
        """
        self.chat_history = []
        self.max_history = max_history
        print(f"Chat memory initialized (max {max_history} exchanges)")

    def add_exchange(self, question, answer):
        """Add a question-answer pair to history"""

        self.chat_history.append((question, answer))

        # Remove oldest if over limit
        if len(self.chat_history) > self.max_history:
            self.chat_history.pop(0)
            print("Oldest exchange removed from memory")

    def get_history(self):
        """Return full chat history as list of tuples"""
        return self.chat_history

    def get_history_as_string(self):
        """Return chat history formatted as readable string"""

        if not self.chat_history:
            return "No previous conversation."

        history_str = ""
        for i, (human, ai) in enumerate(self.chat_history):
            history_str += f"Human: {human}\nAI: {ai}\n\n"

        return history_str.strip()

    def clear(self):
        """Clear all chat history"""
        self.chat_history = []
        print("Chat memory cleared!")

    def get_exchange_count(self):
        """Return number of exchanges stored"""
        return len(self.chat_history)

    def show_history(self):
        """Print full chat history nicely"""

        if not self.chat_history:
            print("No chat history yet.")
            return

        print("\n" + "=" * 50)
        print("CHAT HISTORY")
        print("=" * 50)
        for i, (human, ai) in enumerate(self.chat_history):
            print(f"\nExchange {i+1}:")
            print(f"Human : {human}")
            print(f"AI    : {ai}")
        print("=" * 50)


# ── Quick test ───────────────────────────────────────────
if __name__ == "__main__":

    # Create memory
    memory = ChatMemory(max_history=5)

    # Simulate conversation
    memory.add_exchange(
        "How many sick leaves do I get?",
        "You get 12 days of sick leave per year."
    )
    memory.add_exchange(
        "Can I carry them forward?",
        "No, sick leave cannot be carried forward."
    )
    memory.add_exchange(
        "What about casual leave?",
        "You get 10 days of casual leave per year."
    )

    # Show history
    memory.show_history()

    # Show as string
    print("\nHistory as string:")
    print(memory.get_history_as_string())

    # Show count
    print(f"\nTotal exchanges: {memory.get_exchange_count()}")

    # Clear memory
    memory.clear()
    print(f"After clear: {memory.get_exchange_count()} exchanges")