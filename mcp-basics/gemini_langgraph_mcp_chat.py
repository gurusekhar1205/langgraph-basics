
import asyncio
import sys
from pathlib import Path
from typing import TypedDict, Annotated

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
SERVER_FILE = BASE_DIR / "stdio_server.py"


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


async def main():
    client = MultiServerMCPClient(
        {
            "math": {
                "command": sys.executable,
                "args": [str(SERVER_FILE)],
                "transport": "stdio",
                "cwd": str(BASE_DIR),
            }
        }
    )

    tools = await client.get_tools()

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        temperature=0,
    )
    llm_with_tools = llm.bind_tools(tools)

    async def call_model(state: AgentState):
        response = await llm_with_tools.ainvoke(state["messages"])
        return {"messages": [response]}

    builder = StateGraph(AgentState)

    builder.add_node("llm", call_model)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "llm")

    builder.add_conditional_edges(
        "llm",
        tools_condition,
        {
            "tools": "tools",
            END: END,
        },
    )

    builder.add_edge("tools", "llm")

    graph = builder.compile()

    print("MCP + Gemini + LangGraph agent is ready!")
    print("Type 'exit' to stop.\n")

    while True:
        question = input("You: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Agent stopped.")
            break

        if not question:
            continue

        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=question)]}
        )

        final_message = result["messages"][-1]

        print("\nAgent:")
        if isinstance(final_message.content, str):
            print(final_message.content)
        else:
            for block in final_message.content:
                if isinstance(block, dict) and block.get("type") == "text":
                    print(block["text"])
        print()


if __name__ == "__main__":
    asyncio.run(main())
