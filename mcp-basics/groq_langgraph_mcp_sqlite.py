
import asyncio
import sys
from pathlib import Path
from typing import TypedDict, Annotated

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
SERVER_FILE = BASE_DIR / "stdio_server.py"

# Persistent SQLite database file
DB_PATH = BASE_DIR / "checkpoints.sqlite"


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


async def main():
    # 1. Connect to the MCP server
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
    print("Available MCP tools:", [tool.name for tool in tools])

    # 2. Initialize Groq
    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
    )
    llm_with_tools = llm.bind_tools(tools)

    # 3. Define the LLM node
    async def call_model(state: AgentState):
        response = await llm_with_tools.ainvoke(state["messages"])
        return {"messages": [response]}

    # 4. Build the graph
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

    # 5. Configure a stable conversation ID
    config = {
        "configurable": {
            "thread_id": "groq-sqlite-chat-1"
        }
    }

    print("\nSQLite-backed Groq agent is ready!")
    print("Database:", DB_PATH)
    print("Conversation ID: groq-sqlite-chat-1")
    print("Type 'exit' to stop.\n")

    # 6. Open the persistent database and compile the graph
    async with AsyncSqliteSaver.from_conn_string(
        str(DB_PATH)
    ) as checkpointer:

        graph = builder.compile(checkpointer=checkpointer)

        # 7. Interactive chat loop
        while True:
            question = input("You: ").strip()

            if question.lower() in {"exit", "quit"}:
                print("Agent stopped.")
                break

            if not question:
                continue

            try:
                result = await graph.ainvoke(
                    {
                        "messages": [
                            HumanMessage(content=question)
                        ]
                    },
                    config=config,
                )

                final_message = result["messages"][-1]

                print("\nAgent:")

                if isinstance(final_message.content, str):
                    print(final_message.content)
                else:
                    for block in final_message.content:
                        if (
                            isinstance(block, dict)
                            and block.get("type") == "text"
                        ):
                            print(block["text"])

                print()

            except Exception as exc:
                print(f"\nRequest failed: {exc}\n")


if __name__ == "__main__":
    asyncio.run(main())
