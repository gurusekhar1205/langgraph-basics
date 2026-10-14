
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


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
SERVER_FILE = BASE_DIR / "stdio_server.py"


# 1. Define the graph's state
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


async def main():
    # 2. Connect to the MCP server
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

    # 3. Initialize Groq and bind the tools
    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
    )
    llm_with_tools = llm.bind_tools(tools)

    # 4. Define the model node
    async def call_model(state: AgentState):
        response = await llm_with_tools.ainvoke(state["messages"])
        return {"messages": [response]}

    # 5. Build the graph
    builder = StateGraph(AgentState)

    builder.add_node("llm", call_model)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "llm")

    # Route based on whether the model requests a tool
    builder.add_conditional_edges(
        "llm",
        tools_condition,
        {
            "tools": "tools",
            END: END,
        },
    )

    # After tool execution, return to the model
    builder.add_edge("tools", "llm")

    # 6. Compile the graph
    graph = builder.compile()

    # 7. Run the agent
    question = "What is 125 + 275? Use the add_numbers tool."

    print("\nQuestion:", question)

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=question)]}
    )

    # 8. Display the final answer
    final_message = result["messages"][-1]

    print("\nFinal answer:")

    if isinstance(final_message.content, str):
        print(final_message.content)
    else:
        for block in final_message.content:
            if isinstance(block, dict) and block.get("type") == "text":
                print(block["text"])


if __name__ == "__main__":
    asyncio.run(main())
