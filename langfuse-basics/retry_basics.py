
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy


class State(TypedDict):
    message: str


# Used only to simulate a temporary failure.
attempts = 0


def unreliable_node(state: State):
    global attempts
    attempts += 1

    print(f"Attempt {attempts}")

    if attempts < 2:
        raise ValueError("Temporary simulated failure")

    print("Node succeeded!")
    return {"message": "Operation completed successfully"}


builder = StateGraph(State)

builder.add_node(
    "unreliable",
    unreliable_node,
    retry_policy=RetryPolicy(
        max_attempts=3,
        initial_interval=1.0,
        backoff_factor=2.0,
        jitter=False,
        retry_on=ValueError,
    ),
)

builder.add_edge(START, "unreliable")
builder.add_edge("unreliable", END)

graph = builder.compile()

result = graph.invoke({"message": ""})

print("Final state:", result)
