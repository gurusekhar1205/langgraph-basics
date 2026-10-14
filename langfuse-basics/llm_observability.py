
from dotenv import load_dotenv
from typing_extensions import TypedDict

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langfuse import get_client
from langfuse.langchain import CallbackHandler

load_dotenv()

langfuse = get_client()


class State(TypedDict):
    question: str
    answer: str


# Create the real LLM.
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
)


def llm_node(state: State):
    print("Calling the LLM...")

    response = llm.invoke(
        [
            HumanMessage(
                content=state["question"]
            )
        ]
    )

    # Inspect usage information returned by the model.
    print("\nUsage metadata:")
    print(response.usage_metadata)

    print("\nResponse metadata:")
    print(response.response_metadata)

    return {"answer": response.content}


builder = StateGraph(State)

builder.add_node("llm", llm_node)
builder.add_edge(START, "llm")
builder.add_edge("llm", END)

graph = builder.compile()

handler = CallbackHandler()

try:
    result = graph.invoke(
        {
            "question": (
                "Explain LangGraph in three simple sentences."
            ),
            "answer": "",
        },
        config={"callbacks": [handler]},
    )

    print("\nFinal answer:")
    print(result["answer"])

finally:
    langfuse.flush()
    print("\nLangfuse observations flushed.")
