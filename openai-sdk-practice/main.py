
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

api_key = os.getenv("WISE_GATEWAY_API_KEY")

if not api_key:
    raise ValueError("WISE_GATEWAY_API_KEY is missing from .env")

llm = ChatOpenAI(
    api_key=api_key,
    model="chat-fast",
    base_url="https://wise-gateway.wisseninfotech.com",
)

messages = [
    SystemMessage(
        content="You are a helpful assistant. Answer clearly and briefly."
    ),
    HumanMessage(content="My name is Alex."),
    AIMessage(content="Nice to meet you, Alex!"),
    HumanMessage(content="What is my name?"),
]

response = llm.invoke(messages)

print("Answer:", response.content)
