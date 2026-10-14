from typing import TypedDict

from langgraph.graph import StateGraph, START, END


class ChildState(TypedDict):
    message: str
    processed: bool


def process_message(state: ChildState):
    print("Inside child: process_message")

    return {
        "processed": True
    }


builder = StateGraph(ChildState)

builder.add_node("process", process_message)

builder.add_edge(START, "process")
builder.add_edge("process", END)

# No checkpointer here.
# The parent will provide persistence.
child_graph = builder.compile()