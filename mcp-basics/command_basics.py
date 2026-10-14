
from typing import Literal
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command


class State(TypedDict):
    user_type: str
    message: str


def router(
    state: State,
) -> Command[Literal["admin", "user", "guest"]]:
    user_type = state["user_type"]

    if user_type == "admin":
        return Command(
            update={"message": "Welcome, administrator!"},
            goto="admin",
        )

    elif user_type == "user":
        return Command(
            update={"message": "Welcome, user!"},
            goto="user",
        )

    else:
        return Command(
            update={"message": "Welcome, guest!"},
            goto="guest",
        )


def admin_node(state: State):
    print("Admin dashboard selected.")
    return {}


def user_node(state: State):
    print("User dashboard selected.")
    return {}


def guest_node(state: State):
    print("Guest access selected.")
    return {}


builder = StateGraph(State)

builder.add_node("router", router)
builder.add_node("admin", admin_node)
builder.add_node("user", user_node)
builder.add_node("guest", guest_node)

builder.add_edge(START, "router")

# These nodes finish the graph.
builder.add_edge("admin", END)
builder.add_edge("user", END)
builder.add_edge("guest", END)

graph = builder.compile()

result = graph.invoke({
    "user_type": "admin",
    "message": "",
})

print("Final state:", result)
    