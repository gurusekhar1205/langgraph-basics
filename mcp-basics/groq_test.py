
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load GROQ_API_KEY from the project-root .env file
load_dotenv()

# Initialize the Groq model
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
)

# Send a simple test question
response = llm.invoke("Explain LangGraph in two simple sentences.")

# Print the response
print("Groq response:")

if isinstance(response.content, str):
    print(response.content)
else:
    for block in response.content:
        if isinstance(block, dict) and block.get("type") == "text":
            print(block["text"])
