from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command


class State(TypedDict):
    name: str
    age: int


def collect_information(state: State):

    print("Inside collect_information")

    name = interrupt({
        "question": "What is your name?"
    })

    print("Name received:", name)

    age = interrupt({
        "question": "What is your age?"
    })

    print("Age received:", age)

    return {
        "name": name,
        "age": age
    }


builder = StateGraph(State)

builder.add_node("collect_information", collect_information)

builder.add_edge(START, "collect_information")
builder.add_edge("collect_information", END)

checkpointer = InMemorySaver()

graph = builder.compile(
    checkpointer=checkpointer
)


config = {
    "configurable": {
        "thread_id": "multiple-interrupt-demo-1"
    }
}


print("\n--- FIRST RUN ---")

result = graph.invoke(
    {
        "name": "",
        "age": 0
    },
    config=config
)

print("\nGraph result:")
print(result)

print("\nInterrupt:")
print(result["__interrupt__"])

print("\n--- RESUME NAME ---")

result = graph.invoke(
    Command(resume="Guru"),
    config=config
)

print("\nGraph result:")
print(result)

print("\nInterrupt:")
print(result["__interrupt__"])

print("\n--- RESUME AGE ---")

result = graph.invoke(
    Command(resume=21),
    config=config
)

print("\nFinal result:")
print(result)