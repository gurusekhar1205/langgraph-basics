from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from subgraph_shared_child import child_graph


class SharedState(TypedDict):
    message: str
    result: str


def prepare(state: SharedState):
    print("Inside parent: prepare")

    return {
        "message": state["message"]
    }


def call_child(state: SharedState):
    print("Inside parent: calling child")

    return child_graph.invoke(state)


def final_node(state: SharedState):
    print("Inside parent: final")

    print("Result received from child:", state["result"])

    return {}


builder = StateGraph(SharedState)

builder.add_node("prepare", prepare)
builder.add_node("child", call_child)
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

    print("\nFinal parent result:")
    print(result)