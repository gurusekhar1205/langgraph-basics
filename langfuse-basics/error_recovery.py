
import time
from pathlib import Path
from typing import Literal
from typing_extensions import TypedDict

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langfuse import get_client
from langfuse.langchain import CallbackHandler

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

langfuse = get_client()
handler = CallbackHandler()


class WorkflowState(TypedDict):
    message: str
    status: str
    error: str


def prepare(state: WorkflowState):
    print("1. Preparing workflow...")
    return {
        "message": state["message"],
        "status": "started",
        "error": "",
    }


def process(state: WorkflowState):
    print("2. Processing request...")

    try:
        # Simulate a processing failure
        raise ValueError("Simulated processing failure")

    except Exception as exc:
        print(f"   Processing error: {exc}")
        return {
            "status": "failed",
            "error": str(exc),
        }

    # In a real application, successful processing would return
    # its result and set status to "success".


def route_after_process(
    state: WorkflowState,
) -> Literal["fallback", "finish"]:
    if state["status"] == "failed":
        return "fallback"

    return "finish"


def fallback(state: WorkflowState):
    print("3. Fallback activated!")

    return {
        "message": (
            "The main process failed. "
            "A fallback response was provided."
        ),
        "status": "recovered",
    }


def finish(state: WorkflowState):
    print("4. Finishing workflow...")
    return {
        "message": state["message"],
        "status": state["status"],
    }


builder = StateGraph(WorkflowState)

builder.add_node("prepare", prepare)
builder.add_node("process", process)
builder.add_node("fallback", fallback)
builder.add_node("finish", finish)

builder.add_edge(START, "prepare")
builder.add_edge("prepare", "process")

builder.add_conditional_edges(
    "process",
    route_after_process,
    {
        "fallback": "fallback",
        "finish": "finish",
    },
)

builder.add_edge("fallback", "finish")
builder.add_edge("finish", END)

graph = builder.compile()


def main():
    start = time.perf_counter()

    try:
        result = graph.invoke(
            {
                "message": "Test error recovery",
                "status": "started",
                "error": "",
            },
            config={"callbacks": [handler]},
        )

        print("\nFinal result:", result)

    finally:
        duration = time.perf_counter() - start
        print(f"\nTotal duration: {duration:.3f} seconds")
        langfuse.flush()
        print("Langfuse observations flushed.")


if __name__ == "__main__":
    main()
