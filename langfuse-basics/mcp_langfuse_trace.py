
import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langfuse import get_client
from langfuse.langchain import CallbackHandler

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
MCP_DIR = PROJECT_ROOT / "mcp-basics"

load_dotenv(BASE_DIR / ".env")

langfuse = get_client()


async def main():
    server_python = Path(sys.executable)
    server_file = MCP_DIR / "stdio_server.py"

    if not server_python.exists():
        raise FileNotFoundError(f"MCP Python not found: {server_python}")

    if not server_file.exists():
        raise FileNotFoundError(f"MCP server not found: {server_file}")

    client = MultiServerMCPClient({
        "calculator": {
            "transport": "stdio",
            "command": str(server_python),
            "args": [str(server_file)],
        }
    })

    tools = await client.get_tools()
    print("Available MCP tools:", [tool.name for tool in tools])

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
    )
    llm_with_tools = llm.bind_tools(tools)

    async def assistant(state: MessagesState):
        response = await llm_with_tools.ainvoke(state["messages"])
        return {"messages": [response]}

    builder = StateGraph(MessagesState)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")

    graph = builder.compile()
    handler = CallbackHandler()

    try:
        result = await graph.ainvoke(
            {
                "messages": [
                    HumanMessage(content="What is 125 + 275?")
                ]
            },
            config={"callbacks": [handler]},
        )

        print("\nFinal answer:")
        print(result["messages"][-1].content)

    finally:
        langfuse.flush()
        print("\nLangfuse observations flushed.")


if __name__ == "__main__":
    asyncio.run(main())
