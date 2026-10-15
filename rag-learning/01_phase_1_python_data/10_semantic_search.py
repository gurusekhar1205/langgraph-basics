
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

DATA_DIR = Path(__file__).parent / "data" / "sample_docs"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 80
CHUNK_OVERLAP = 20

TOP_K = 3


# --------------------------------------------------
# 2. Load the embedding model
# --------------------------------------------------

print("Loading embedding model...")

model = SentenceTransformer(MODEL_NAME)


# --------------------------------------------------
# 3. Configure the text splitter
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
        print(f"Skipping empty document: {file_path.name}")
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
# 5. Create chunks and preserve metadata
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
            }
        )

print("Chunks created:", len(chunks))


# --------------------------------------------------
# 6. Generate document embeddings
# --------------------------------------------------

document_texts = [chunk["text"] for chunk in chunks]

document_embeddings = model.encode(
    document_texts,
    normalize_embeddings=True,
    show_progress_bar=True,
)

print("Document embedding shape:", document_embeddings.shape)


# --------------------------------------------------
# 7. Define the semantic search function
# --------------------------------------------------

def semantic_search(query, top_k=3):
    """
    Return the top_k chunks ranked by cosine similarity
    to the user's query.
    """

    if not query.strip():
        raise ValueError("Query cannot be empty.")

    if not chunks:
        return []

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    # Embed the user's query.
    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
    )

    # Compare the query vector with every document vector.
    # Both sets of vectors are normalized, so dot product
    # equals cosine similarity.
    scores = document_embeddings @ query_embedding

    # Sort chunk indices from highest score to lowest score.
    ranked_indices = np.argsort(scores)[::-1]

    # Return no more than the number of available chunks.
    result_count = min(top_k, len(chunks))

    results = []

    for index in ranked_indices[:result_count]:
        chunk = chunks[int(index)]

        results.append(
            {
                "chunk_id": chunk["chunk_id"],
                "source": chunk["source"],
                "text": chunk["text"],
                "score": float(scores[index]),
            }
        )

    return results


# --------------------------------------------------
# 8. Run a search
# --------------------------------------------------

query = "How can I regain access to my account?"

print("\nUser query:", query)
print("\nTop search results")
print("=" * 60)

results = semantic_search(query, top_k=TOP_K)

for rank, result in enumerate(results, start=1):
    print(f"\nRank: {rank}")
    print(f"Chunk ID: {result['chunk_id']}")
    print(f"Source: {result['source']}")
    print(f"Similarity score: {result['score']:.4f}")
    print(f"Text: {result['text']}")
    print("-" * 60)


# --------------------------------------------------
# 9. Try another query
# --------------------------------------------------

second_query = "What should I do if the application crashes?"

print("\nSecond query:", second_query)

second_results = semantic_search(second_query, top_k=TOP_K)

for rank, result in enumerate(second_results, start=1):
    print(
        f"{rank}. [{result['score']:.4f}] "
        f"{result['source']} | {result['text']}"
    )
