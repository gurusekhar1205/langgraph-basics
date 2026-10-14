
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START
from langgraph.graph import MessagesState
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

api_key = os.getenv("WISE_GATEWAY_API_KEY")
if not api_key:
    raise ValueError("WISE_GATEWAY_API_KEY is missing from .env")

llm = ChatOpenAI(
    api_key=api_key,
    model="chat-fast",
    base_url="https://wise-gateway.wisseninfotech.com",
)

def call_model(state: MessagesState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

builder = StateGraph(MessagesState)
builder.add_node("chat", call_model)
builder.add_edge(START, "chat")

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "conversation-1"}}

while True:
    user_input = input("You: ").strip()

    if user_input.lower() == "exit":
        break

    if not user_input:
        continue

    result = graph.invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        config=config,
    )

    print("Assistant:", result["messages"][-1].content)
