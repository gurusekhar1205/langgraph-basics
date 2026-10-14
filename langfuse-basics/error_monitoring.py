
import time
from pathlib import Path
from typing_extensions import TypedDict

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langfuse import get_client
from langfuse.langchain import CallbackHandler

# Load environment variables
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

langfuse = get_client()
handler = CallbackHandler()


class WorkflowState(TypedDict):
    message: str


def prepare(state: WorkflowState):
    print("Preparing workflow...")
    return {"message": state["message"] + " | prepared"}


def failing_node(state: WorkflowState):
    print("Executing failing node...")

    # Deliberately trigger an error for this experiment
    raise ValueError("Demo error: simulated processing failure")


def finish(state: WorkflowState):
    print("Finishing workflow...")
    return {"message": state["message"] + " | finished"}


# Build the graph
builder = StateGraph(WorkflowState)

builder.add_node("prepare", prepare)
builder.add_node("failing_node", failing_node)
builder.add_node("finish", finish)

builder.add_edge(START, "prepare")
builder.add_edge("prepare", "failing_node")
builder.add_edge("failing_node", "finish")
builder.add_edge("finish", END)

graph = builder.compile()


def main():
    start = time.perf_counter()

    try:
        result = graph.invoke(
            {"message": "Error monitoring experiment"},
            config={"callbacks": [handler]},
        )
        print("Final state:", result)

    except Exception as exc:
        print(f"\nCaught error: {type(exc).__name__}: {exc}")

    finally:
        duration = time.perf_counter() - start
        print(f"Total execution duration: {duration:.3f} seconds")

        langfuse.flush()
        print("Langfuse observations flushed.")


if __name__ == "__main__":
    main()
