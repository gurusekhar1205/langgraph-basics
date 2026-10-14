from typing import TypedDict

from langgraph.graph import StateGraph, START, END


class ChildState(TypedDict):
    message: str
    result: str


def process_message(state: ChildState):
    print("Inside child graph")

    return {
        "result": f"Child processed: {state['message']}"
    }


builder = StateGraph(ChildState)

builder.add_node("process_message", process_message)

builder.add_edge(START, "process_message")
builder.add_edge("process_message", END)

child_graph = builder.compile()


if __name__ == "__main__":

    result = child_graph.invoke({
        "message": "Hello from parent",
        "result": ""
    })

    print("\nChild graph result:")
    print(result)