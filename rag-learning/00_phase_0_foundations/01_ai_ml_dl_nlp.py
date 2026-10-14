"""
Lesson 1: AI, ML, Deep Learning, NLP, Generative AI, and LLMs

Purpose:
1. Demonstrate a simple rule-based program.
2. Show why manually written rules are different from learned models.
3. Introduce the relationship between AI concepts.
4. Run without third-party dependencies.
"""

from pprint import pprint


def classify_support_message(message: str) -> str:
    """
    A small rule-based classifier.

    IMPORTANT:
    This is NOT a machine learning model.
    The conditions below were written manually by the developer.
    """
    text = message.lower()

    if "password" in text or "login" in text:
        return "Account Support"

    if "payment" in text or "invoice" in text:
        return "Billing Support"

    if "error" in text or "crash" in text:
        return "Technical Support"

    return "General Support"


def main() -> None:
    print("=" * 60)
    print("LESSON 1: AI FOUNDATIONS")
    print("=" * 60)

    messages = [
        "I cannot reset my password",
        "My invoice contains an incorrect amount",
        "The application crashes when I open it",
        "I want to know more about the product",
    ]

    print("\n1. RULE-BASED CLASSIFICATION")
    print("-" * 60)

    for message in messages:
        category = classify_support_message(message)

        print(f"Message  : {message}")
        print(f"Category : {category}")
        print()

    print("\n2. WHAT RULE-BASED PROGRAMMING MEANS")
    print("-" * 60)

    print("The developer manually wrote conditions such as:")
    print("  - If the message contains 'password', select Account Support.")
    print("  - If the message contains 'invoice', select Billing Support.")
    print()
    print("The program follows these predefined conditions.")
    print("It does not learn these rules from training examples.")

    print("\n3. LIMITATIONS OF THESE RULES")
    print("-" * 60)

    difficult_messages = [
        "I am locked out of my account",
        "Money was deducted twice",
        "The software unexpectedly stopped working",
    ]

    for message in difficult_messages:
        category = classify_support_message(message)

        print(f"Message  : {message}")
        print(f"Category : {category}")
        print()

    print("Inspect the predictions above.")
    print("A message can express an issue without containing the")
    print("specific keywords anticipated by the developer.")
    print()
    print("This illustrates one limitation of simple keyword rules.")
    print("It does NOT prove that every machine learning model")
    print("will classify these messages correctly.")

    print("\n4. HOW THE AI CONCEPTS RELATE")
    print("-" * 60)

    concepts = {
        "AI": "The broad field of intelligent computational systems",
        "ML": "Methods that learn patterns from data",
        "Deep Learning": "ML based on multilayer neural networks",
        "NLP": "Methods for processing human language",
        "Generative AI": "AI that generates new content",
        "LLM": "A large language model used for language tasks",
        "RAG": "Retrieval of external evidence for generation",
    }

    pprint(concepts, sort_dicts=False)

    print("\n5. IMPORTANT DISTINCTION")
    print("-" * 60)
    print("This script demonstrates predefined rules.")
    print("It does not train a neural network or call an LLM.")
    print("We will introduce actual embedding and model APIs")
    print("in later lessons, after learning the fundamentals.")


if __name__ == "__main__":
    main()
