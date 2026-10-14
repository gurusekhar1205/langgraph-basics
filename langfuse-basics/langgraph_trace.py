
import time

from dotenv import load_dotenv
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

load_dotenv()

langfuse = get_client()


class State(TypedDict):
    question: str
    context: str
    answer: str


def retrieve_node(state: State):
    start_time = time.perf_counter()

    print("Retrieving documents...")
    time.sleep(1)  # Simulate retrieval taking 1 second

    context = "Check gateway connectivity and configuration."

    elapsed = time.perf_counter() - start_time
    print(f"Retrieval latency: {elapsed:.3f} seconds")

    return {"context": context}


def answer_node(state: State):
    start_time = time.perf_counter()

    print("Preparing answer...")
    time.sleep(2)  # Simulate answer preparation taking 2 seconds

    answer = (
        f"Question: {state['question']}\n"
        f"Retrieved context: {state['context']}\n"
        "Suggested action: Check gateway connectivity and configuration."
    )

    elapsed = time.perf_counter() - start_time
    print(f"Answer latency: {elapsed:.3f} seconds")

    return {"answer": answer}


builder = StateGraph(State)

builder.add_node("retrieve", retrieve_node)
builder.add_node("answer", answer_node)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "answer")
builder.add_edge("answer", END)

graph = builder.compile()

handler = CallbackHandler()

result = graph.invoke(
    {
        "question": "Claude gateway was unreachable",
        "context": "",
        "answer": "",
    },
    config={"callbacks": [handler]},
)

print("\nFinal answer:")
print(result["answer"])

langfuse.flush()
print("\nTrace sent to Langfuse.")
