
from dotenv import load_dotenv
from langfuse import get_client

load_dotenv()

langfuse = get_client()

# Verify that the credentials work.
if not langfuse.auth_check():
    raise SystemExit("Langfuse authentication failed.")

print("Langfuse authentication successful!")

# Create a trace for one troubleshooting request.
with langfuse.start_as_current_observation(
    as_type="span",
    name="troubleshooting-request",
) as trace:

    trace.update(
        input={
            "question": "Claude gateway was unreachable"
        }
    )

    # Simulate RAG retrieval.
    with langfuse.start_as_current_observation(
        as_type="span",
        name="retrieve-documents",
    ) as retrieval:

        retrieval.update(
            input={"query": "Claude gateway unreachable"},
            output={"chunks_found": 3},
        )

    # Simulate answer generation.
    with langfuse.start_as_current_observation(
        as_type="span",
        name="prepare-answer",
    ) as answer:

        answer.update(
            input={"chunks_found": 3},
            output="Check gateway connectivity and configuration.",
        )

    trace.update(
        output="Troubleshooting answer prepared."
    )

# Send pending observations to Langfuse.
langfuse.flush()

print("Trace submitted to Langfuse!")
