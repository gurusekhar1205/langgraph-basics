import asyncio

from mcp import Client


async def main():

    async with Client("http://localhost:8000/mcp") as client:

        print("Connected to MCP server")

        result = await client.list_tools()

        print("\nAvailable tools:")

        for tool in result.tools:
            print("-", tool.name)

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