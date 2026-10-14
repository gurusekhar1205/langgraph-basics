
import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import HumanMessage, ToolMessage


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
    # 1. Discover MCP tools
    tools = await client.get_tools()
    print("Available tools:", [tool.name for tool in tools])

    # 2. Initialize Gemini and provide the available tools
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        temperature=0,
    )
    llm_with_tools = llm.bind_tools(tools)

    # 3. Ask a question
    messages = [
        HumanMessage(
            content="What is 25 + 35? Use the add_numbers tool to calculate it."
        )
    ]

    # 4. Ask Gemini to decide whether to call a tool
    ai_message = await llm_with_tools.ainvoke(messages)
    messages.append(ai_message)

    # 5. Execute every tool requested by Gemini
    for tool_call in ai_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_call_id = tool_call["id"]

        print(f"\nExecuting tool: {tool_name}")
        print("Arguments:", tool_args)

        # Find the matching LangChain tool
        selected_tool = next(
            (tool for tool in tools if tool.name == tool_name),
            None,
        )

        if selected_tool is None:
            tool_result = f"Error: Unknown tool {tool_name}"
        else:
            try:
                raw_result = await selected_tool.ainvoke(tool_args)

                # MCP results may be returned as content blocks
                if isinstance(raw_result, list):
                    text_parts = [
                        block["text"]
                        for block in raw_result
                        if isinstance(block, dict)
                        and block.get("type") == "text"
                    ]
                    tool_result = "\n".join(text_parts) or str(raw_result)
                else:
                    tool_result = str(raw_result)

            except Exception as exc:
                tool_result = f"Tool execution failed: {exc}"

        print("Tool result:", tool_result)

        # 6. Return the tool result to the conversation
        messages.append(
            ToolMessage(
                content=tool_result,
                tool_call_id=tool_call_id,
            )
        )

    # 7. Ask Gemini to produce the final answer
    final_response = await llm_with_tools.ainvoke(messages)

    print("\nFinal answer:")
    for block in (
        final_response.content
        if isinstance(final_response.content, list)
        else []
    ):
        if isinstance(block, dict) and block.get("type") == "text":
            print(block["text"])

    if isinstance(final_response.content, str):
        print(final_response.content)


if __name__ == "__main__":
    asyncio.run(main())
