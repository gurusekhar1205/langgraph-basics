from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from subgraph_native_child import child_graph


class State(TypedDict):
    message: str
    result: str


def prepare(state: State):
    print("Inside parent: prepare")

    return {}


def final_node(state: State):
    print("Inside parent: final")
    print("Child result:", state["result"])

    return {}


builder = StateGraph(State)

builder.add_node("prepare", prepare)

# Add the compiled child graph directly
builder.add_node("child", child_graph)

builder.add_node("final", final_node)

builder.add_edge(START, "prepare")
builder.add_edge("prepare", "child")
builder.add_edge("child", "final")
builder.add_edge("final", END)

parent_graph = builder.compile()


if __name__ == "__main__":

    result = parent_graph.invoke({
        "message": "Hello from parent",
        "result": ""
    })

    print("\nFinal result:")
    print(result)