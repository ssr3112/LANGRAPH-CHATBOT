# tool_backend.py

import sqlite3
from typing import TypedDict, Annotated, List

from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

# ------------------- 1. LLM (Gemini) -------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
)

# ------------------- 2. Tools -------------------
search_tool = DuckDuckGoSearchRun(region="us-en")

@tool
def improve_answer(question: str, answer: str) -> str:
    """
    Improve an interview answer: make it structured, clear, and concise.
    """
    prompt = (
        f"Question: {question}\n\n"
        f"Candidate answer:\n{answer}\n\n"
        "Rewrite this answer to be clear, structured, and impressive for a tech interview. "
        "Keep it under 200 words."
    )
    return llm.invoke(prompt).content

@tool
def create_flashcards(text: str) -> list[dict]:
    """
    Turn notes into flashcards: each with 'question' and 'answer'.
    """
    prompt = (
        "Create 5 flashcards from the following text.\n"
        "Return as lines in the format: Q: ... | A: ...\n\n"
        + text
    )
    content = llm.invoke(prompt).content
    cards: list[dict] = []
    for line in content.splitlines():
        if "Q:" in line and "A:" in line:
            q_part, a_part = line.split("A:", 1)
            question = q_part.replace("Q:", "").strip()
            answer = a_part.strip()
            if question and answer:
                cards.append({"question": question, "answer": answer})
    return cards

tools = [search_tool, improve_answer, create_flashcards]
llm_with_tools = llm.bind_tools(tools)

# ------------------- 3. State -------------------
class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

# ------------------- 4. Nodes -------------------
def chat_node(state: ChatState) -> ChatState:
    """
    LLM node that may answer directly or request a tool call.
    """
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)

# ------------------- 5. Checkpointer (SQLite) -------------------
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# ------------------- 6. Graph -------------------
graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")
graph.add_edge("chat_node", END)

# Compile LangGraph app
chatbot = graph.compile(checkpointer=checkpointer)

# ------------------- 7. Helper: list all threads -------------------
def retrieve_all_threads() -> list[str]:
    all_threads: set[str] = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)
