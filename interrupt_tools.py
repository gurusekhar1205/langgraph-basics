from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command


class State(TypedDict):
    action: str
    result: str


def delete_file_tool(state: State):
    print("Inside delete_file_tool")

    approval = interrupt({
        "question": "Do you approve this sensitive operation?",
        "action": state["action"],
        "warning": "This operation cannot be undone."
    })

    print("Human approval:", approval)

    if approval:
        return {
            "result": "File deleted successfully."
        }

    return {
        "result": "File deletion rejected by human."
    }


builder = StateGraph(State)

builder.add_node("delete_file", delete_file_tool)

builder.add_edge(START, "delete_file")
builder.add_edge("delete_file", END)

checkpointer = InMemorySaver()

graph = builder.compile(
    checkpointer=checkpointer
)


config = {
    "configurable": {
        "thread_id": "tool-interrupt-demo-1"
    }
}


print("\n--- FIRST RUN ---")

result = graph.invoke(
    {
        "action": "Delete important.txt",
        "result": ""
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