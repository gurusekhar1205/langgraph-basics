from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from subgraph_persistence_child import child_graph


class ParentState(TypedDict):
    message: str
    processed: bool


def prepare(state: ParentState):
    print("Inside parent: prepare")

    return {}


def final_node(state: ParentState):
    print("Inside parent: final")
    print("Processed:", state["processed"])

    return {}


builder = StateGraph(ParentState)

builder.add_node("prepare", prepare)

# Child graph becomes a node
builder.add_node("child", child_graph)

builder.add_node("final", final_node)

builder.add_edge(START, "prepare")
builder.add_edge("prepare", "child")
builder.add_edge("child", "final")
builder.add_edge("final", END)


checkpointer = InMemorySaver()

parent_graph = builder.compile(
    checkpointer=checkpointer
)


config = {
    "configurable": {
        "thread_id": "subgraph-persistence-1"
    }
}


if __name__ == "__main__":

    result = parent_graph.invoke(
        {
            "message": "Hello",
            "processed": False
        },
        config=config
    )

    print("\nFinal result:")
    print(result)