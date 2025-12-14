# tool_frontend_interview.py

import uuid
import streamlit as st

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from tool_backend import chatbot, retrieve_all_threads

# ======================= Helpers =======================

def extract_text(content) -> str:
    """
    Normalize Gemini content to a plain string.
    Handles cases like [{'type': 'text', 'text': '...'}].
    """
    # Case 1: list of parts (Gemini-style)
    if isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                texts.append(part["text"])
            else:
                texts.append(str(part))
        return "".join(texts)
    # Case 2: already a simple string
    if isinstance(content, str):
        return content
    # Fallback
    return str(content)

def generate_thread_id() -> str:
    return str(uuid.uuid4())

def add_thread(thread_id: str):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []

def load_conversation(thread_id: str):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])

# =================== Session Initialization ===================

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

add_thread(st.session_state["thread_id"])

# ============================ Sidebar ============================

st.sidebar.title("Interview Prep Assistant (LangGraph)")

if st.sidebar.button("🆕 New Chat"):
    reset_chat()

st.sidebar.header("My Conversations")
for thread_id in st.session_state["chat_threads"][::-1]:
    if st.sidebar.button(thread_id):
        st.session_state["thread_id"] = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            else:
                # Ignore ToolMessages in history
                continue
            temp_messages.append(
                {"role": role, "content": extract_text(msg.content)}
            )
        st.session_state["message_history"] = temp_messages

# ============================ Main UI ============================

st.title("💼 Interview Assistant with Tools")
st.caption("Gemini + LangGraph: web search, improve answers, create flashcards")

# Render history
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input(
    "Ask: mock interview question, paste your answer, or notes to make flashcards"
)

if user_input:
    # Show user message
    st.session_state["message_history"].append(
        {"role": "user", "content": user_input}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {"thread_id": st.session_state["thread_id"]},
        "run_name": "chat_turn",
    }

    # Assistant streaming
    with st.chat_message("assistant"):
        status_holder = {"box": None}

        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                # Show when a tool runs
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    label = f"🔧 Using `{tool_name}` …"
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(label, expanded=True)
                    else:
                        status_holder["box"].update(
                            label=label,
                            state="running",
                            expanded=True,
                        )

                # Stream ONLY assistant text
                if isinstance(message_chunk, AIMessage):
                    yield extract_text(message_chunk.content)

        ai_message = st.write_stream(ai_only_stream())

        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tools finished",
                state="complete",
                expanded=False,
            )

    # Save assistant message as plain text
    st.session_state["message_history"].append(
        {"role": "assistant", "content": extract_text(ai_message)}
    )

