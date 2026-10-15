
from pathlib import Path
from pprint import pprint

from langchain_text_splitters import RecursiveCharacterTextSplitter


DATA_DIR = (
    Path(__file__).parent
    / "data"
    / "sample_docs"
)

CHUNK_SIZE = 80
CHUNK_OVERLAP = 20


def load_documents(folder: Path) -> list[dict]:
    """Load text files and preserve their source metadata."""

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

    # Create the recursive text splitter.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )

    all_chunks = []

    print("=" * 60)
    print("LESSON 4: RECURSIVE CHARACTER TEXT SPLITTING")
    print("=" * 60)

    print(f"Documents loaded: {len(documents)}")
    print(f"Chunk size: {CHUNK_SIZE} characters")
    print(f"Chunk overlap: {CHUNK_OVERLAP} characters")

    for document in documents:
        # Split the text into smaller pieces.
        text_chunks = splitter.split_text(document["text"])

        print(f"\nSource: {document['source']}")
        print(f"Chunks created: {len(text_chunks)}")

        for index, text_chunk in enumerate(text_chunks):
            chunk_record = {
                "chunk_id": (
                    f"{document['id']}_chunk_{index:03d}"
                ),
                "document_id": document["id"],
                "source": document["source"],
                "chunk_index": index,
                "text": text_chunk,
            }

            all_chunks.append(chunk_record)

            print(f"\n  Chunk {index}")
            print(f"  ID: {chunk_record['chunk_id']}")
            print(f"  Characters: {len(text_chunk)}")
            print(f"  Text: {text_chunk!r}")

    print("\n" + "=" * 60)
    print("CHUNKING SUMMARY")
    print("=" * 60)

    print(f"Total documents: {len(documents)}")
    print(f"Total chunks: {len(all_chunks)}")

    # Validate the generated chunks.
    assert all(chunk["text"] for chunk in all_chunks)
    assert all(chunk["source"] for chunk in all_chunks)
    assert all(
        len(chunk["text"]) <= CHUNK_SIZE
        for chunk in all_chunks
    )

    print("\nSUCCESS: Chunks and source metadata validated.")

    print("\nFirst chunk record:")
    pprint(all_chunks[0] if all_chunks else None)


if __name__ == "__main__":
    main()
