
from pathlib import Path
from pprint import pprint

# Locate the sample documents.
DATA_DIR = (
    Path(__file__).parent
    / "data"
    / "sample_docs"
)

CHUNK_SIZE = 80
CHUNK_OVERLAP = 20


def chunk_text(
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[str]:
    """Split text into overlapping character-based chunks."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be >= 0 and < chunk_size"
        )

    text = text.strip()

    if not text:
        return []

    chunks = []
    step = chunk_size - overlap

    for start in range(0, len(text), step):
        chunk = text[start : start + chunk_size]
        chunks.append(chunk)

        if start + chunk_size >= len(text):
            break

    return chunks


def load_documents(folder: Path) -> list[dict]:
    """Read text files and preserve their source names."""

    documents = []

    for path in sorted(folder.glob("*.txt")):
        documents.append(
            {
                "id": path.stem,
                "text": path.read_text(encoding="utf-8").strip(),
                "source": path.name,
            }
        )

    return documents


def main():
    documents = load_documents(DATA_DIR)
    all_chunks = []

    print("=" * 60)
    print("LESSON 3: DOCUMENT CHUNKING")
    print("=" * 60)

    print(f"Documents loaded: {len(documents)}")
    print(f"Chunk size: {CHUNK_SIZE} characters")
    print(f"Chunk overlap: {CHUNK_OVERLAP} characters")

    for document in documents:
        chunks = chunk_text(
            document["text"],
            CHUNK_SIZE,
            CHUNK_OVERLAP,
        )

        print(f"\nSource: {document['source']}")
        print(f"Chunks created: {len(chunks)}")

        for index, chunk in enumerate(chunks):
            chunk_record = {
                "chunk_id": f"{document['id']}_chunk_{index:03d}",
                "document_id": document["id"],
                "source": document["source"],
                "chunk_index": index,
                "text": chunk,
            }

            all_chunks.append(chunk_record)

            print(f"\n  Chunk {index}")
            print(f"  ID: {chunk_record['chunk_id']}")
            print(f"  Characters: {len(chunk)}")
            print(f"  Text: {chunk!r}")

    print("\n" + "=" * 60)
    print("CHUNKING SUMMARY")
    print("=" * 60)
    print(f"Total documents: {len(documents)}")
    print(f"Total chunks: {len(all_chunks)}")

    # Basic validation.
    assert all(chunk["text"] for chunk in all_chunks)
    assert all(chunk["source"] for chunk in all_chunks)

    print("\nSUCCESS: All chunks preserve their source metadata.")

    print("\nFirst chunk record:")
    pprint(all_chunks[0] if all_chunks else None)


if __name__ == "__main__":
    main()
