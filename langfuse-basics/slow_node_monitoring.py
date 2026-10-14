
import time
from pathlib import Path

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langfuse import get_client
from langfuse.langchain import CallbackHandler

from typing_extensions import TypedDict

# Load environment variables
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

langfuse = get_client()
handler = CallbackHandler()


# Define the graph state
class WorkflowState(TypedDict):
    message: str


# Node 1: Fast operation
def prepare(state: WorkflowState):
    print("Preparing request...")
    return {"message": state["message"] + " | prepared"}


# Node 2: Deliberately slow operation
def slow_process(state: WorkflowState):
    print("Starting slow processing...")

    time.sleep(2)  # Simulate a slow operation

    print("Slow processing completed.")
    return {"message": state["message"] + " | processed"}


# Node 3: Fast operation
def finish(state: WorkflowState):
    print("Finishing workflow...")
    return {"message": state["message"] + " | finished"}


# Build the graph
builder = StateGraph(WorkflowState)

builder.add_node("prepare", prepare)
builder.add_node("slow_process", slow_process)
builder.add_node("finish", finish)

builder.add_edge(START, "prepare")
builder.add_edge("prepare", "slow_process")
builder.add_edge("slow_process", "finish")
builder.add_edge("finish", END)

graph = builder.compile()


def main():
    start = time.perf_counter()

    result = graph.invoke(
        {"message": "Latency experiment"},
        config={"callbacks": [handler]},
    )

    duration = time.perf_counter() - start

    print("\nFinal state:", result)
    print(f"Total graph duration: {duration:.3f} seconds")

    langfuse.flush()
    print("Langfuse observations flushed.")


if __name__ == "__main__":
    main()
