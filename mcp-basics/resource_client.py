import asyncio

from mcp import Client, StdioServerParameters


server = StdioServerParameters(
    command="python",
    args=["resource_server.py"],
)


async def main():

    async with Client(server) as client:

        print("Connected to MCP server")

        result = await client.list_resources()

        print("\nAvailable resources:")

        for resource in result.resources:
            print("-", resource.uri)

        resource = await client.read_resource(
            "config://application"
        )

        print("\nResource content:")
        print(resource)


if __name__ == "__main__":
    asyncio.run(main())