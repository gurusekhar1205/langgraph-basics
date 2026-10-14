
import time
from pathlib import Path
from typing_extensions import TypedDict

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langfuse import get_client
from langfuse.langchain import CallbackHandler


# Load environment variables from .env
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

langfuse = get_client()
handler = CallbackHandler()


# Define the workflow state
class MonitoringState(TypedDict):
    request: str
    result: str
    status: str


# Node 1: Receive the request
def receive_request(state: MonitoringState):
    print("1. Receiving request...")

    return {
        "request": state["request"],
        "result": "",
        "status": "running",
    }


# Node 2: Process the request
def process_request(state: MonitoringState):
    print("2. Processing request...")

    # Simulate a slow operation to test latency monitoring
    time.sleep(3)

    return {
        "result": f"Processed: {state['request']}",
        "status": "success",
    }


# Node 3: Finish the workflow
def finish(state: MonitoringState):
    print("3. Finishing workflow...")

    return {
        "result": state["result"],
        "status": state["status"],
    }


# Build the LangGraph workflow
builder = StateGraph(MonitoringState)

builder.add_node("receive_request", receive_request)
builder.add_node("process_request", process_request)
builder.add_node("finish", finish)

builder.add_edge(START, "receive_request")
builder.add_edge("receive_request", "process_request")
builder.add_edge("process_request", "finish")
builder.add_edge("finish", END)

graph = builder.compile()


# Execute and monitor the workflow
def main():
    start = time.perf_counter()

    try:
        result = graph.invoke(
            {
                "request": "Test Langfuse latency alert",
                "result": "",
                "status": "new",
            },
            config={"callbacks": [handler]},
        )

        duration = time.perf_counter() - start

        print("\n--- Monitoring Summary ---")
        print("Final result:", result)
        print(f"Workflow latency: {duration:.3f} seconds")
        print(f"Workflow status: {result['status']}")

    except Exception as exc:
        print(f"\nWorkflow failed: {type(exc).__name__}: {exc}")

    finally:
        langfuse.flush()
        print("Langfuse observations flushed.")


if __name__ == "__main__":
    main()
