from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from subgraph_persistence_interrupt_child import child_graph


class ParentState(TypedDict):
    message: str
    approved: bool
    result: str


def prepare(state: ParentState):
    print("Parent: prepare")

    return {
        "message": "Deploy application",
        "approved": False,
        "result": ""
    }


def final_node(state: ParentState):
    print("Parent: final")

    return {
        "result": "Parent completed successfully"
    }


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
        "thread_id": "subgraph-interrupt-1"
    }
}


print("\n--- FIRST RUN ---")

result = parent_graph.invoke(
    {
        "message": "",
        "approved": False,
        "result": ""
    },
    config
)

print("\nGraph paused:")
print(result)


print("\n--- RESUMING ---")

result = parent_graph.invoke(
    Command(resume=True),
    config
)

print("\nFinal result:")
print(result)