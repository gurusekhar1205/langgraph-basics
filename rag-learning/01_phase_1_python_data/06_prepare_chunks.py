
from pathlib import Path
from pprint import pprint

from langchain_text_splitters import RecursiveCharacterTextSplitter


# --------------------------------------------------
# 1. Configure paths and chunking settings
# --------------------------------------------------

DATA_DIR = Path(__file__).parent / "data" / "sample_docs"

CHUNK_SIZE = 80
CHUNK_OVERLAP = 20


# --------------------------------------------------
# 2. Create the text splitter
# --------------------------------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    separators=["\n\n", "\n", " ", ""],
)


# --------------------------------------------------
# 3. Load source documents
# --------------------------------------------------

documents = []

for file_path in sorted(DATA_DIR.glob("*.txt")):
    text = file_path.read_text(encoding="utf-8")
    text = " ".join(text.split())

    if not text:
        print(f"Skipping empty document: {file_path.name}")
        continue

    documents.append(
        {
            "document_id": file_path.stem,
            "source": file_path.name,
            "text": text,
        }
    )

print("Source documents loaded:", len(documents))


# --------------------------------------------------
# 4. Split documents and attach metadata
# --------------------------------------------------

chunks = []

for document in documents:
    text_chunks = splitter.split_text(document["text"])

    for chunk_index, chunk_text in enumerate(text_chunks):
        chunk_record = {
            "chunk_id": (
                f"{document['document_id']}_chunk_{chunk_index:03d}"
            ),
            "document_id": document["document_id"],
            "chunk_index": chunk_index,
            "source": document["source"],
            "text": chunk_text,
            "char_count": len(chunk_text),
        }

        chunks.append(chunk_record)


# --------------------------------------------------
# 5. Display the prepared chunks
# --------------------------------------------------

print("Total chunks created:", len(chunks))
print()

for chunk in chunks:
    print("=" * 60)
    pprint(chunk)


# --------------------------------------------------
# 6. Validate the output
# --------------------------------------------------

chunk_ids = [chunk["chunk_id"] for chunk in chunks]

assert len(chunk_ids) == len(set(chunk_ids)), "Duplicate chunk IDs found"

assert all(chunk["text"].strip() for chunk in chunks), (
    "An empty chunk was found"
)

assert all(chunk["char_count"] <= CHUNK_SIZE for chunk in chunks), (
    "A chunk exceeds the configured size"
)

assert all(
    chunk["document_id"] and chunk["source"]
    for chunk in chunks
), "Required metadata is missing"

print()
print("Validation successful.")
print("Every chunk has text, a unique ID, and source metadata.")
