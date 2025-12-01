import streamlit as st
from streamlit_backend import chatbot          # your existing working backend
from langchain_core.messages import HumanMessage, AIMessage
import uuid
import time

# ---------- Utility functions ----------

def generate_thread_id() -> str:
    """Generate a new unique thread id."""
    return str(uuid.uuid4())

def add_thread(thread_id: str):
    """Register a thread id in the session list if not present."""
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)

def reset_chat():
    """Start a completely new chat thread."""
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []

def load_conversation(thread_id: str):
    """Load conversation from LangGraph checkpoint for given thread id."""
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    # state.values is a dict; "messages" may or may not exist
    messages = state.values.get("messages", [])
    temp_messages = []

    for msg in messages:
        if isinstance(msg, HumanMessage):
            role = "user"
        elif isinstance(msg, AIMessage):
            role = "assistant"
        else:
            # fallback in case of system/other messages
            role = "assistant"
        temp_messages.append({"role": role, "content": msg.content})

    st.session_state["message_history"] = temp_messages

# ---------- Session setup ----------

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = []

# ensure current thread_id is registered
add_thread(st.session_state["thread_id"])

# ---------- Sidebar UI (threads) ----------

st.sidebar.title("LangGraph Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()

st.sidebar.header("My Conversations")

# show latest threads first
for thread_id in st.session_state["chat_threads"][::-1]:
    if st.sidebar.button(thread_id):
        st.session_state["thread_id"] = thread_id
        load_conversation(thread_id)

# ---------- Main chat UI ----------

# show history for current thread
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

user_input = st.chat_input("Type here")

if user_input:
    # config with current thread_id (this is what makes resume work)
    CONFIG = {"configurable": {"thread_id": st.session_state["thread_id"]}}

    # add user message to history
    st.session_state["message_history"].append(
        {"role": "user", "content": user_input}
    )
    with st.chat_message("user"):
        st.text(user_input)

    # assistant with typing effect (same as your previous working code)
    assistant_container = st.chat_message("assistant")
    with assistant_container:
        typing_placeholder = st.empty()
        typing_placeholder.markdown("**🤖 AI is typing...**")

        # call LangGraph backend (Gemini) with thread-aware config
        response = chatbot.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=CONFIG,
        )
        full_response = response["messages"][-1].content

        # simulate typing
        typing_placeholder.markdown("")
        for i in range(0, len(full_response), 2):
            partial_response = full_response[: i + 2]
            typing_placeholder.markdown(partial_response)
            time.sleep(0.03)

        typing_placeholder.markdown(full_response)

    # store assistant reply in history
    st.session_state["message_history"].append(
        {"role": "assistant", "content": full_response}
    )
