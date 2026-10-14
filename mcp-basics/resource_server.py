from mcp.server import MCPServer

mcp = MCPServer("Resource Demo Server")


@mcp.resource("config://application")
def application_config() -> str:
    """Application configuration information."""
    return """
Application: AI Troubleshooting Agent
Environment: Development
Version: 1.0
Status: Running
"""
    

if __name__ == "__main__":
    mcp.run()