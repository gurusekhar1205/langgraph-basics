
import time
from pathlib import Path
from typing import Literal

from typing_extensions import TypedDict
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy
from langfuse import get_client
from langfuse.langchain import CallbackHandler

# Load environment variables
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

langfuse = get_client()
handler = CallbackHandler()


# Define workflow state
class WorkflowState(TypedDict):
    message: str
    status: str
    error: str


# Node 1: Prepare the workflow
def prepare(state: WorkflowState):
    print("1. Preparing workflow...")

    return {
        "message": state["message"],
        "status": "started",
        "error": "",
    }


# Node 2: Process with retries

attempt_count = 0

def process(state: WorkflowState):
    global attempt_count
    attempt_count += 1

    print(f"2. Processing request... Attempt {attempt_count}")

    raise ValueError(
        f"Simulated processing failure on attempt {attempt_count}"
    )



# Node 3: Fallback after retries are exhausted
def fallback(state: WorkflowState):
    print("3. Fallback activated!")

    return {
        "message": (
            "The main process failed after retries. "
            "A fallback response was provided."
        ),
        "status": "recovered",
    }


# Node 4: Finish the workflow
def finish(state: WorkflowState):
    print("4. Finishing workflow...")

    return {
        "message": state["message"],
        "status": state["status"],
    }


# Decide where to go after processing
def route_after_process(
    state: WorkflowState,
) -> Literal["fallback", "finish"]:
    if state["status"] == "failed":
        return "fallback"

    return "finish"


builder = StateGraph(WorkflowState)

builder.add_node("prepare", prepare)

# Retry the process node up to 3 attempts
builder.add_node(
    "process",
    process,
    retry_policy=RetryPolicy(
        max_attempts=3,
        retry_on=lambda exc: isinstance(exc, ValueError),
        initial_interval=0.1,
        jitter=False,
    ),
)

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
                "message": "Retry experiment",
                "status": "started",
                "error": "",
            },
            config={"callbacks": [handler]},
        )

        print("\nFinal result:", result)

    except Exception as exc:
        print(f"\nWorkflow failed after retries: {exc}")

    finally:
        duration = time.perf_counter() - start
        print(f"\nTotal duration: {duration:.3f} seconds")

        langfuse.flush()
        print("Langfuse observations flushed.")


if __name__ == "__main__":
    main()
