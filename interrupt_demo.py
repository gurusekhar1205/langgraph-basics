from typing import TypedDict, Literal

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command


class State(TypedDict):
    action: str
    approved: bool


def approval_node(state: State) -> Command[Literal["proceed", "cancel"]]:

    print("Inside approval_node")

    approved = interrupt({
        "question": "Do you approve this action?",
        "action": state["action"],
        "details": "This operation requires human approval."
    })

    print("Human response:", approved)

    if approved:
        return Command(goto="proceed")

    return Command(goto="cancel")


def proceed_node(state: State):

    print("Action APPROVED")

    return {
        "approved": True
    }


def cancel_node(state: State):

    print("Action REJECTED")

    return {
        "approved": False
    }


builder = StateGraph(State)

builder.add_node("approval", approval_node)
builder.add_node("proceed", proceed_node)
builder.add_node("cancel", cancel_node)

builder.add_edge(START, "approval")

builder.add_edge("proceed", END)
builder.add_edge("cancel", END)

checkpointer = InMemorySaver()

graph = builder.compile(
    checkpointer=checkpointer
)


config = {
    "configurable": {
        "thread_id": "approval-demo-1"
    }
}


print("\n--- FIRST RUN ---")

result = graph.invoke(
    {
        "action": "Delete a file",
        "approved": False
    },
    config=config
)

print("\nGraph result:")
print(result)


print("\n--- RESUMING WITH APPROVAL ---")

result = graph.invoke(
    Command(resume=True),
    config=config
)

print("\nFinal result:")
print(result)