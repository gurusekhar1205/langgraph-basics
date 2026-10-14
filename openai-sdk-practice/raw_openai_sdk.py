
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("WISE_GATEWAY_API_KEY")

if not api_key:
    raise ValueError("WISE_GATEWAY_API_KEY is missing from .env")

# Create the official SDK client
client = OpenAI(
    api_key=api_key,
    base_url="https://wise-gateway.wisseninfotech.com",
)

# Send a request using the SDK directly
response = client.chat.completions.create(
    model="chat-fast",
    messages=[
        {
            "role": "user",
            "content": "Explain an API in two simple sentences."
        }
    ],
)

# Extract and print the generated text
print("Answer:")
print(response.choices[0].message.content)

# Inspect token usage
if response.usage:
    print("\nToken usage:")
    print("Input:", response.usage.prompt_tokens)
    print("Output:", response.usage.completion_tokens)
    print("Total:", response.usage.total_tokens)
