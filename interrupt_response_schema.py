from typing import TypedDict, Any

from pydantic import BaseModel

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command


# Human's expected response
class Decision(BaseModel):
    approved: bool
    note: str | None = None


class State(TypedDict):
    answer: Any


def review_node(state: State):

    print("Inside review_node")

    decision = interrupt(
        {
            "question": "Do you approve this action?",
            "action": "Deploy application"
        },
        response_schema=Decision
    )

    print("Human decision:", decision)
    print("Type:", type(decision))

    return {
        "answer": decision
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
        "thread_id": "schema-demo-1"
    }
}


print("\n--- FIRST RUN ---")

result = graph.invoke(
    {
        "answer": None
    },
    config=config
)

print("\nGraph result:")
print(result)

print("\nInterrupt:")
print(result["__interrupt__"])


print("\n--- RESUMING ---")

result = graph.invoke(
    Command(
        resume={
            "approved": True,
            "note": "Everything looks good."
        }
    ),
    config=config
)

print("\nFinal result:")
print(result)