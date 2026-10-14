from typing import TypedDict

from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    message: str
    result: str


def process_message(state: State):
    print("Inside native child graph")

    return {
        "result": f"Child processed: {state['message']}"
    }


builder = StateGraph(State)

builder.add_node("process_message", process_message)

builder.add_edge(START, "process_message")
builder.add_edge("process_message", END)

child_graph = builder.compile()