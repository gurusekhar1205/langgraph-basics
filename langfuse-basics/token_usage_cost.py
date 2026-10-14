
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langfuse import get_client
from langfuse.langchain import CallbackHandler

# Load environment variables from this project's .env file
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# Initialize Langfuse
langfuse = get_client()

# Initialize Groq model
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
)

# Attach Langfuse tracing
handler = CallbackHandler()


def main():
    messages = [
        SystemMessage(
            content="You are a helpful Python tutor. Answer briefly."
        ),
        HumanMessage(
            content="Explain Python lists in three simple sentences."
        ),
    ]

    response = llm.invoke(
        messages,
        config={"callbacks": [handler]},
    )

    print("\nModel response:")
    print(response.content)

    print("\nToken usage metadata:")
    print(response.usage_metadata)

    usage = response.usage_metadata or {}

    print("\nInput tokens:", usage.get("input_tokens", "Not reported"))
    print("Output tokens:", usage.get("output_tokens", "Not reported"))
    print("Total tokens:", usage.get("total_tokens", "Not reported"))

    print("\nProvider-specific response metadata:")
    print(response.response_metadata.get("token_usage", "Not available"))

    langfuse.flush()
    print("\nLangfuse observations flushed.")


if __name__ == "__main__":
    main()
