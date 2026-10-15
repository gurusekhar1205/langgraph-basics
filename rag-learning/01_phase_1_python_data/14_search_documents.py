
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).parent
DB_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "troubleshooting_documents"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3


# --------------------------------------------------
# 2. Connect to the existing persistent collection
# --------------------------------------------------

client = chromadb.PersistentClient(path=str(DB_DIR))

try:
    collection = client.get_collection(name=COLLECTION_NAME)
except Exception as exc:
    raise RuntimeError(
        "The ChromaDB collection was not found. "
        "Run 14_ingest_documents.py first."
    ) from exc

if collection.count() == 0:
    raise RuntimeError(
        "The collection is empty. Run 14_ingest_documents.py first."
    )

print(f"Collection: {COLLECTION_NAME}")
print(f"Stored records: {collection.count()}")


# --------------------------------------------------
# 3. Load the query embedding model
# --------------------------------------------------

model = SentenceTransformer(MODEL_NAME)


# --------------------------------------------------
# 4. Embed the query and retrieve stored chunks
# --------------------------------------------------

def search_documents(query, top_k=TOP_K):
    query_embedding = model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    matches = []

    for index, chunk_id in enumerate(results["ids"][0]):
        distance = results["distances"][0][index]

        matches.append({
            "id": chunk_id,
            "text": results["documents"][0][index],
            "source": results["metadatas"][0][index]["source"],
            "chunk_index": results["metadatas"][0][index]["chunk_index"],
            "distance": distance,
            "similarity": 1.0 - distance,
        })

    return matches


# --------------------------------------------------
# 5. Search with user queries
# --------------------------------------------------

queries = [
    "How can I regain access to my account?",
    "What should I do if the application crashes?",
    "How do I handle duplicate payments?",
]

for query in queries:
    print("\n" + "=" * 65)
    print(f"Query: {query}")
    print("=" * 65)

    matches = search_documents(query)

    for rank, match in enumerate(matches, start=1):
        print(f"\nRank: {rank}")
        print(f"Chunk ID: {match['id']}")
        print(f"Source: {match['source']}")
        print(f"Cosine distance: {match['distance']:.4f}")
        print(f"Approx. cosine similarity: {match['similarity']:.4f}")
        print(f"Text: {match['text']}")
