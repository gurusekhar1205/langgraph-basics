
import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
SERVER_FILE = BASE_DIR / "stdio_server.py"

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


async def main():
    # 1. Discover the MCP tools
    tools = await client.get_tools()
    print("Available MCP tools:", [tool.name for tool in tools])

    # 2. Initialize Groq
    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
    )

    # 3. Give Groq access to the tool definitions
    llm_with_tools = llm.bind_tools(tools)

    # 4. Ask Groq to use the calculator
    question = "What is 125 + 275? Use the add_numbers tool."

    response = await llm_with_tools.ainvoke(question)

    print("\nModel tool calls:")
    print(response.tool_calls)

    if not response.tool_calls:
        print("\nModel response:")
        print(response.content)
        return

    # 5. Execute the tool requested by Groq
    for tool_call in response.tool_calls:
        selected_tool = next(
            (tool for tool in tools if tool.name == tool_call["name"]),
            None,
        )

        if selected_tool is None:
            print("Unknown tool:", tool_call["name"])
            return

        result = await selected_tool.ainvoke(tool_call["args"])
        print("\nTool result:", result)

        # 6. Send the tool result back to the model
        from langchain_core.messages import HumanMessage, ToolMessage

        messages = [
            HumanMessage(content=question),
            response,
            ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"],
            ),
        ]

        final_response = await llm_with_tools.ainvoke(messages)

        print("\nFinal answer:")
        print(final_response.content)


if __name__ == "__main__":
    asyncio.run(main())
