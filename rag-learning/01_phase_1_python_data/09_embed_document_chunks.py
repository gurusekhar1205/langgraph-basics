
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


# --------------------------------------------------
# 1. Configure the project
# --------------------------------------------------

DATA_DIR = Path(__file__).parent / "data" / "sample_docs"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 80
CHUNK_OVERLAP = 20


# --------------------------------------------------
# 2. Load the embedding model
# --------------------------------------------------

print("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)


# --------------------------------------------------
# 3. Create the text splitter
# --------------------------------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    separators=["\n\n", "\n", " ", ""],
)


# --------------------------------------------------
# 4. Load and clean source documents
# --------------------------------------------------

documents = []

for file_path in sorted(DATA_DIR.glob("*.txt")):
    text = file_path.read_text(encoding="utf-8")
    text = " ".join(text.split())

    if not text:
        print(f"Skipping empty file: {file_path.name}")
        continue

    documents.append(
        {
            "document_id": file_path.stem,
            "source": file_path.name,
            "text": text,
        }
    )

print("Documents loaded:", len(documents))


# --------------------------------------------------
# 5. Split documents and preserve metadata
# --------------------------------------------------

chunks = []

for document in documents:
    text_chunks = splitter.split_text(document["text"])

    for chunk_index, chunk_text in enumerate(text_chunks):
        chunks.append(
            {
                "chunk_id": (
                    f"{document['document_id']}_chunk_{chunk_index:03d}"
                ),
                "document_id": document["document_id"],
                "chunk_index": chunk_index,
                "source": document["source"],
                "text": chunk_text,
                "char_count": len(chunk_text),
            }
        )

print("Chunks created:", len(chunks))


# --------------------------------------------------
# 6. Generate an embedding for every chunk
# --------------------------------------------------

texts = [chunk["text"] for chunk in chunks]

embeddings = model.encode(
    texts,
    normalize_embeddings=True,
    show_progress_bar=True,
)

print("Embedding matrix shape:", embeddings.shape)


# --------------------------------------------------
# 7. Attach each vector to its corresponding chunk
# --------------------------------------------------

for chunk, embedding in zip(chunks, embeddings):
    chunk["embedding"] = embedding

    # Store a readable vector summary for inspection.
    # We do not print all 384 values.
    chunk["embedding_dimensions"] = len(embedding)
    chunk["embedding_norm"] = float(np.linalg.norm(embedding))


# --------------------------------------------------
# 8. Inspect the results
# --------------------------------------------------

print("\nPrepared embedded chunks")
print("=" * 60)

for chunk in chunks:
    print(f"\nChunk ID: {chunk['chunk_id']}")
    print(f"Source: {chunk['source']}")
    print(f"Text: {chunk['text']}")
    print(f"Character count: {chunk['char_count']}")
    print(f"Embedding dimensions: {chunk['embedding_dimensions']}")
    print(f"Embedding norm: {chunk['embedding_norm']:.4f}")
    print("First 5 embedding values:", chunk["embedding"][:5])


# --------------------------------------------------
# 9. Validate the embedded dataset
# --------------------------------------------------

assert len(chunks) == len(embeddings)
assert all(len(chunk["embedding"]) == 384 for chunk in chunks)
assert len({chunk["chunk_id"] for chunk in chunks}) == len(chunks)
assert all(chunk["source"] and chunk["text"] for chunk in chunks)

print("\nValidation successful.")
print("Every chunk has text, metadata, and a 384-dimensional embedding.")
