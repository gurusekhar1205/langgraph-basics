
from pathlib import Path
from pprint import pprint

# 1. Locate the folder containing our documents.
DATA_DIR = (
    Path(__file__).parent
    / "data"
    / "sample_docs"
)

# 2. Find all .txt files in that folder.
file_paths = sorted(DATA_DIR.glob("*.txt"))

print("=" * 60)
print("LESSON 2: LOADING TEXT DOCUMENTS")
print("=" * 60)

print(f"\nData directory: {DATA_DIR}")
print(f"Number of files found: {len(file_paths)}")

# 3. Read each file and create a document record.
documents = []

for file_path in file_paths:
    text = file_path.read_text(encoding="utf-8")

    document = {
        "id": file_path.stem,
        "text": text.strip(),
        "source": file_path.name,
        "source_path": str(file_path),
    }

    documents.append(document)

# 4. Display each document and its statistics.
print("\n1. LOADED DOCUMENTS")
print("-" * 60)

for document in documents:
    text = document["text"]

    print(f"\nDocument ID: {document['id']}")
    print(f"Source: {document['source']}")
    print(f"Characters: {len(text)}")
    print(f"Words: {len(text.split())}")
    print(f"Text: {text}")

# 5. Display the complete collection.
print("\n2. COMPLETE DOCUMENT COLLECTION")
print("-" * 60)

pprint(documents, sort_dicts=False)

# 6. Validate that loading worked.
assert len(documents) == 3, "Expected exactly 3 documents"
assert all(document["text"] for document in documents)

print("\nSUCCESS: All three documents were loaded.")
