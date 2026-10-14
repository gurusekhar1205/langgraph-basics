from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from subgraph_child import child_graph


class ParentState(TypedDict):
    user_message: str
    child_result: str


def prepare_message(state: ParentState):
    print("Inside parent: prepare_message")

    return {
        "user_message": state["user_message"]
    }


def call_child(state: ParentState):
    print("Inside parent: calling child graph")

    result = child_graph.invoke({
        "message": state["user_message"],
        "result": ""
    })

    return {
        "child_result": result["result"]
    }


def final_node(state: ParentState):
    print("Inside parent: final_node")

    return {
        "child_result": state["child_result"]
    }


builder = StateGraph(ParentState)

builder.add_node("prepare", prepare_message)
builder.add_node("child", call_child)
builder.add_node("final", final_node)

builder.add_edge(START, "prepare")
builder.add_edge("prepare", "child")
builder.add_edge("child", "final")
builder.add_edge("final", END)

parent_graph = builder.compile()


if __name__ == "__main__":

    result = parent_graph.invoke({
        "user_message": "Hello from parent",
        "child_result": ""
    })

    print("\nParent graph result:")
    print(result)