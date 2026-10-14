from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command


class State(TypedDict):
    content: str


def review_node(state: State):

    print("Inside review_node")

    edited_content = interrupt({
        "instruction": "Review and edit this content",
        "content": state["content"]
    })

    print("Human provided:", edited_content)

    return {
        "content": edited_content
    }


builder = StateGraph(State)

builder.add_node("review", review_node)

builder.add_edge(START, "review")
builder.add_edge("review", END)


checkpointer = InMemorySaver()

graph = builder.compile(
    checkpointer=checkpointer
)


config = {
    "configurable": {
        "thread_id": "review-demo-1"
    }
}


print("\n--- FIRST RUN ---")

result = graph.invoke(
    {
        "content": "LangGraph is a framework for AI applications."
    },
    config=config
)

print("\nGraph result:")
print(result)


print("\n--- RESUMING WITH EDITED CONTENT ---")

result = graph.invoke(
    Command(
        resume="LangGraph is a framework for building stateful AI workflows."
    ),
    config=config
)

print("\nFinal result:")
print(result)