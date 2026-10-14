import asyncio

from mcp import Client, StdioServerParameters


server = StdioServerParameters(
    command="python",
    args=["server.py"],
)


async def main():

    async with Client(server) as client:

        print("Connected to MCP server")

        # Discover tools
        result = await client.list_tools()

        print("\nAvailable tools:")

        for tool in result.tools:
            print("-", tool.name)

        # Call add_numbers
        result = await client.call_tool(
            "add_numbers",
            {
                "a": 10,
                "b": 20,
            },
        )

        print("\nTool result:")
        print(result)


if __name__ == "__main__":
    asyncio.run(main())