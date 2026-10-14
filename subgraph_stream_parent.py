from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from subgraph_stream_child import child_graph


class State(TypedDict):
    message: str
    result: str


def prepare(state: State):
    print("Parent: prepare")

    return {}


def final_node(state: State):
    print("Parent: final")

    return {}


builder = StateGraph(State)

builder.add_node("prepare", prepare)
builder.add_node("child", child_graph)
builder.add_node("final", final_node)

builder.add_edge(START, "prepare")
builder.add_edge("prepare", "child")
builder.add_edge("child", "final")
builder.add_edge("final", END)

parent_graph = builder.compile()


if __name__ == "__main__":

    print("\n--- STREAMING WITHOUT SUBGRAPHS ---")

    for chunk in parent_graph.stream(
        {
            "message": "START",
            "result": ""
        },
        stream_mode="updates",
        subgraphs=True
    ):
        print(chunk)