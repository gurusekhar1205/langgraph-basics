from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command


class State(TypedDict):
    age: int
    valid: bool


def ask_age(state: State):
    print("Inside ask_age")

    age = interrupt({
        "question": "Please enter your age:"
    })

    print("Human entered:", age)

    return {
        "age": age
    }


def validate_age(state: State):
    print("Inside validate_age")

    age = state["age"]

    if isinstance(age, int) and 0 <= age <= 120:
        print("Age is valid")
        return {"valid": True}

    print("Age is invalid")
    return {"valid": False}


def route_after_validation(state: State):
    if state["valid"]:
        return "finish"

    return "ask_age"


def finish(state: State):
    print("Final valid age:", state["age"])
    return {}


builder = StateGraph(State)

builder.add_node("ask_age", ask_age)
builder.add_node("validate_age", validate_age)
builder.add_node("finish", finish)

builder.add_edge(START, "ask_age")
builder.add_edge("ask_age", "validate_age")

builder.add_conditional_edges(
    "validate_age",
    route_after_validation,
    {
        "ask_age": "ask_age",
        "finish": "finish"
    }
)

builder.add_edge("finish", END)

checkpointer = InMemorySaver()

graph = builder.compile(
    checkpointer=checkpointer
)


config = {
    "configurable": {
        "thread_id": "validation-demo-1"
    }
}


print("\n--- FIRST RUN ---")

result = graph.invoke(
    {
        "age": 0,
        "valid": False
    },
    config=config
)

print("\nGraph result:")
print(result)

print("\n--- RESUME WITH INVALID AGE ---")

result = graph.invoke(
    Command(resume=-5),
    config=config
)

print("\nGraph result:")
print(result)

print("\n--- RESUME WITH VALID AGE ---")

result = graph.invoke(
    Command(resume=21),
    config=config
)

print("\nFinal result:")
print(result)