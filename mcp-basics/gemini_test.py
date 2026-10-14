import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


# Load variables from .env
load_dotenv()


# Create Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0,
)


# Send a message
response = llm.invoke("Explain MCP in one sentence.")


print("Gemini response:")

for block in response.content:
    if block.get("type") == "text":
        print(block["text"])