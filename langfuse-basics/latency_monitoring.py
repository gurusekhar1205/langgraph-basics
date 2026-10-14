
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langfuse import get_client
from langfuse.langchain import CallbackHandler

# 1. Load environment variables
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# 2. Initialize Langfuse and Groq
langfuse = get_client()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
)

handler = CallbackHandler()


def main():
    # Experiment A: Measure a normal Python operation
    print("\n--- Experiment A: Python operation ---")

    start_time = time.perf_counter()

    time.sleep(2)

    end_time = time.perf_counter()
    elapsed = end_time - start_time

    print(f"Python operation duration: {elapsed:.3f} seconds")

    # Experiment B: Measure the Groq model call
    print("\n--- Experiment B: Groq LLM call ---")

    messages = [
        HumanMessage(
            content="Explain latency in one simple sentence."
        )
    ]

    start_time = time.perf_counter()

    response = llm.invoke(
        messages,
        config={"callbacks": [handler]},
    )

    end_time = time.perf_counter()
    llm_duration = end_time - start_time

    print("\nModel response:")
    print(response.content)

    print(f"\nLLM call duration: {llm_duration:.3f} seconds")

    # 3. Flush pending trace data
    langfuse.flush()
    print("\nLangfuse observations flushed.")


if __name__ == "__main__":
    main()
