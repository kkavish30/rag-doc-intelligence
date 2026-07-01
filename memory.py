from langchain_classic.memory import ConversationBufferMemory
from langchain.messages import HumanMessage, AIMessage

class SessionMemory:

    def __init__(self):
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            human_prefix="User",
            ai_prefix="Assistant"
        )


    def add_turn(self, user_message: str, assistant_message: str):
        self.memory.chat_memory.add_user_message(user_message)
        self.memory.chat_memory.add_ai_message(assistant_message)


    def get_history(self) -> list[dict]:
        messages = self.memory.chat_memory.messages
        history = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                history.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                history.append({"role": "assistant", "content": msg.content})
        
        return history
    

    def get_history_as_string(self) -> str:
        history = self.get_history()
        if not history:
            return "No previous conversation."
        
        lines = []
        for msg in history:
            prefix = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{prefix}: {msg["content"]}")
        return "\n".join(lines)
    

    def clear(self):
        self.memory.clear()


    def turn_count(self) -> int:
        return len(self.memory.chat_memory.messages) // 2
    

if __name__ == "__main__":
    mem = SessionMemory()
    mem.add_turn("What is attention?", "Attention is a mechanism that lets a model focus on relevant parts of the input.")
    mem.add_turn("How is it different from convolution?", "Convolution applies fixed local filters, while attention computes dynamic weights based on content.")

    print("History as messages: ")
    for m in mem.get_history():
        print(f"{m["role"]}: {m["content"][:50]}")

    print(f"\nTurn count: {mem.turn_count()}")
    
    print("\nHistory as strings: ")
    print(mem.get_history_as_string())