"""
Lesson 1: Document Basics for RAG

Learn how to represent, inspect, clean, and organize
text documents before preparing them for retrieval.
"""

from pprint import pprint


# --------------------------------------------------
# 1. REPRESENTING DOCUMENTS
# --------------------------------------------------

documents = [
    {
        "id": "doc_001",
        "text": "Users can reset a forgotten password from the login page.",
        "source": "account_guide.md",
        "category": "account",
    },
    {
        "id": "doc_002",
        "text": "Billing issues include duplicate payments and incorrect invoices.",
        "source": "billing_guide.md",
        "category": "billing",
    },
    {
        "id": "doc_003",
        "text": "If the application crashes, collect the error message and logs.",
        "source": "troubleshooting.md",
        "category": "technical",
    },
]


print("=" * 60)
print("LESSON 1: DOCUMENT BASICS FOR RAG")
print("=" * 60)


# --------------------------------------------------
# 2. INSPECTING A DOCUMENT
# --------------------------------------------------

print("\n1. INSPECTING ONE DOCUMENT")
print("-" * 60)

first_document = documents[0]

print("Document ID :", first_document["id"])
print("Text        :", first_document["text"])
print("Source      :", first_document["source"])
print("Category    :", first_document["category"])


# --------------------------------------------------
# 3. WHY METADATA MATTERS
# --------------------------------------------------

print("\n2. DOCUMENT METADATA")
print("-" * 60)

for document in documents:
    print(
        f'{document["id"]} | '
        f'{document["category"]} | '
        f'{document["source"]}'
    )

print("\nMetadata describes the document.")
print("It can help us filter and trace retrieved information.")


# --------------------------------------------------
# 4. CLEANING TEXT
# --------------------------------------------------

print("\n3. CLEANING A DOCUMENT")
print("-" * 60)

messy_text = """
    Users   can reset a forgotten password.

    Contact support if the reset link expires.
"""

cleaned_text = " ".join(messy_text.split())

print("Original text:")
print(repr(messy_text))

print("\nCleaned text:")
print(repr(cleaned_text))


# --------------------------------------------------
# 5. COUNTING CHARACTERS AND WORDS
# --------------------------------------------------

print("\n4. BASIC TEXT STATISTICS")
print("-" * 60)

for document in documents:
    text = document["text"]

    print(f'Document: {document["id"]}')
    print(f"Characters: {len(text)}")
    print(f"Words: {len(text.split())}")
    print()


# --------------------------------------------------
# 6. BUILDING A SIMPLE SEARCH BASELINE
# --------------------------------------------------

print("5. SIMPLE KEYWORD SEARCH")
print("-" * 60)

query = "password reset"
query_words = set(query.lower().split())

for document in documents:
    document_words = set(document["text"].lower().split())
    matching_words = query_words.intersection(document_words)

    print(f'Document: {document["id"]}')
    print(f"Matching words: {sorted(matching_words)}")
    print(f"Match count: {len(matching_words)}")
    print()


# --------------------------------------------------
# 7. IMPORTANT LIMITATIONS
# --------------------------------------------------

print("6. IMPORTANT LIMITATIONS")
print("-" * 60)

print("This exercise does not use embeddings or an LLM.")
print("Keyword overlap is not the same as semantic similarity.")
print("Real documents may require parsing, cleaning, and chunking.")


# --------------------------------------------------
# 8. DISPLAY THE DOCUMENT COLLECTION
# --------------------------------------------------

print("\n7. COMPLETE DOCUMENT COLLECTION")
print("-" * 60)

pprint(documents, sort_dicts=False)
