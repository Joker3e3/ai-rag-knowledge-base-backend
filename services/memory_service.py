from langchain.memory import ConversationBufferWindowMemory

user_memories = {}

def get_user_memory(user_id: str):
    if user_id not in user_memories:
        user_memories[user_id] = ConversationBufferWindowMemory(
            memory_key="chat_history",
            input_key="question",
            output_key="answer",
            k=5,
            return_messages=True
        )

    return user_memories[user_id]