import asyncio
import sys
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient


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

    print("Connecting to MCP server...")

    # Discover MCP tools
    tools = await client.get_tools()

    print("\nAvailable LangChain tools:")

    for tool in tools:
        print("-", tool.name)

    # Get the add_numbers tool
    add_tool = next(
        tool for tool in tools
        if tool.name == "add_numbers"
    )

    print("\nCalling add_numbers...")

    # Execute the tool
    result = await add_tool.ainvoke(
        {
            "a": 10,
            "b": 20
        }
    )

    print("\nResult:")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())