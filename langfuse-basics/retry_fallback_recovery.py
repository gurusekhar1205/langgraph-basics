
import time
from pathlib import Path
from typing_extensions import TypedDict

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy
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
    raise ValueError("Simulated processing failure")


def fallback(state: WorkflowState):
    print("3. Fallback activated!")
    return {
        "message": (
            "The main process failed after retries. "
            "A fallback response was provided."
        ),
        "status": "recovered",
        "error": state["error"],
    }


def finish(state: WorkflowState):
    print("4. Finishing recovery...")
    return {
        "message": state["message"],
        "status": state["status"],
        "error": state["error"],
    }


# Graph 1: Main processing workflow
main_builder = StateGraph(WorkflowState)
main_builder.add_node("prepare", prepare)
main_builder.add_node(
    "process",
    process,
    retry_policy=RetryPolicy(
        max_attempts=3,
        retry_on=lambda exc: isinstance(exc, ValueError),
        initial_interval=0.1,
        jitter=False,
    ),
)
main_builder.add_edge(START, "prepare")
main_builder.add_edge("prepare", "process")
main_graph = main_builder.compile()


# Graph 2: Recovery workflow
recovery_builder = StateGraph(WorkflowState)
recovery_builder.add_node("fallback", fallback)
recovery_builder.add_node("finish", finish)
recovery_builder.add_edge(START, "fallback")
recovery_builder.add_edge("fallback", "finish")
recovery_builder.add_edge("finish", END)
recovery_graph = recovery_builder.compile()


def main():
    start = time.perf_counter()

    initial_state = {
        "message": "Retry and fallback experiment",
        "status": "started",
        "error": "",
    }

    try:
        # Run the main graph
        result = main_graph.invoke(
            initial_state,
            config={"callbacks": [handler]},
        )
        print("\nMain workflow result:", result)

    except Exception as exc:
        # All retry attempts have failed
        print(f"\nMain workflow failed after retries: {exc}")

        # Run the fallback graph with the error preserved
        recovery_state = {
            **initial_state,
            "status": "failed",
            "error": str(exc),
        }

        result = recovery_graph.invoke(
            recovery_state,
            config={"callbacks": [handler]},
        )

        print("\nFinal recovered result:", result)

    finally:
        duration = time.perf_counter() - start
        print(f"\nTotal duration: {duration:.3f} seconds")
        langfuse.flush()
        print("Langfuse observations flushed.")


if __name__ == "__main__":
    main()
