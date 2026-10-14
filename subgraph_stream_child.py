from typing import TypedDict
import time
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    message: str
    result: str


def step_one(state: State):
    print("Child: step_one")
    time.sleep(3)
    return {
        "message": state["message"] + " → step1"
    }


def step_two(state: State):
    print("Child: step_two")
    time.sleep(3)
    return {
        "message": state["message"] + " → step2"
    }


def step_three(state: State):
    print("Child: step_three")
    time.sleep(3)
    return {
        "result": state["message"] + " → step3"
    }


builder = StateGraph(State)

builder.add_node("step_one", step_one)
builder.add_node("step_two", step_two)
builder.add_node("step_three", step_three)

builder.add_edge(START, "step_one")
builder.add_edge("step_one", "step_two")
builder.add_edge("step_two", "step_three")
builder.add_edge("step_three", END)

child_graph = builder.compile()