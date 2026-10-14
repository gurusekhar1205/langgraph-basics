
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


# 1. Define the conversation state
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


async def main():
    # 2. Connect to the MCP server and discover its tools
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
    print("Available tools:", [tool.name for tool in tools])

    # 3. Initialize Gemini with tool definitions
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        temperature=0,
    )
    llm_with_tools = llm.bind_tools(tools)

    # 4. Define the LLM node
    async def call_model(state: AgentState):
        response = await llm_with_tools.ainvoke(state["messages"])
        return {"messages": [response]}

    # 5. Build the graph
    builder = StateGraph(AgentState)

    builder.add_node("llm", call_model)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "llm")

    # Route to tools if Gemini requests a tool call;
    # otherwise, finish.
    builder.add_conditional_edges(
        "llm",
        tools_condition,
        {
            "tools": "tools",
            END: END,
        },
    )

    # After executing a tool, ask Gemini again
    builder.add_edge("tools", "llm")

    graph = builder.compile()

    # 6. Run the graph
    question = "What is 25 + 35? Use the add_numbers tool to calculate it."

    print("\nQuestion:", question)
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=question)]}
    )

    
    print("\nFinal answer:")
    final_message = result["messages"][-1]
    
    if isinstance(final_message.content, str):
        print(final_message.content)
    else:
        for block in final_message.content:
            if isinstance(block, dict) and block.get("type") == "text":
                print(block["text"])



if __name__ == "__main__":
    asyncio.run(main())
