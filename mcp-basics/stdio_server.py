from mcp.server.fastmcp import FastMCP

# Create MCP server
mcp = FastMCP("LangChain MCP Server")


# Create an MCP tool
@mcp.tool()
def add_numbers(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


# Start the MCP server
if __name__ == "__main__":
    mcp.run()