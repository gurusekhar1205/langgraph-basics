from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt


class ChildState(TypedDict):
    message: str
    approved: bool
    result: str


def child_step_1(state: ChildState):
    print("Child: step 1")

    decision = interrupt({
        "question": "Do you want to continue?",
        "message": state["message"]
    })

    print("Child received:", decision)

    return {
        "approved": decision
    }


def child_step_2(state: ChildState):
    print("Child: step 2")

    return {
        "result": f"Child completed: {state['message']}"
    }


builder = StateGraph(ChildState)

builder.add_node("child_step_1", child_step_1)
builder.add_node("child_step_2", child_step_2)

builder.add_edge(START, "child_step_1")
builder.add_edge("child_step_1", "child_step_2")
builder.add_edge("child_step_2", END)

child_graph = builder.compile()