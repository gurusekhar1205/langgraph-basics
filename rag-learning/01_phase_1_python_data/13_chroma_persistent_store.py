
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "sample_docs"

# Chroma stores its database files here.
DB_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "troubleshooting_documents"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3


# --------------------------------------------------
# 2. Load and chunk documents
# --------------------------------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=80,
    chunk_overlap=20,
    separators=["\n\n", "\n", " ", ""],
)

chunks = []

for file_path in sorted(DATA_DIR.glob("*.txt")):
    text = file_path.read_text(encoding="utf-8")
    text = " ".join(text.split())

    split_texts = splitter.split_text(text)
    source_stem = file_path.stem

    for index, chunk_text in enumerate(split_texts):
        chunks.append({
            "id": f"{source_stem}_chunk_{index:03d}",
            "source": file_path.name,
            "chunk_index": index,
            "text": chunk_text,
        })

print(f"Documents chunked: {len(list(DATA_DIR.glob('*.txt')))}")
print(f"Chunks prepared: {len(chunks)}")


# --------------------------------------------------
# 3. Load the embedding model
# --------------------------------------------------

model = SentenceTransformer(MODEL_NAME)


# --------------------------------------------------
# 4. Generate document embeddings
# --------------------------------------------------

chunk_texts = [chunk["text"] for chunk in chunks]

embeddings = model.encode(
    chunk_texts,
    convert_to_numpy=True,
    normalize_embeddings=True,
)

print(f"Embedding shape: {embeddings.shape}")


# --------------------------------------------------
# 5. Connect to persistent Chroma storage
# --------------------------------------------------

client = chromadb.PersistentClient(path=str(DB_DIR))

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)

print(f"Collection: {collection.name}")
print(f"Records before upsert: {collection.count()}")


# --------------------------------------------------
# 6. Store chunks, vectors, and metadata
# --------------------------------------------------

collection.upsert(
    ids=[chunk["id"] for chunk in chunks],
    documents=[chunk["text"] for chunk in chunks],
    embeddings=embeddings.tolist(),
    metadatas=[
        {
            "source": chunk["source"],
            "chunk_index": chunk["chunk_index"],
        }
        for chunk in chunks
    ],
)

print(f"Records after upsert: {collection.count()}")
print(f"Persistent database path: {DB_DIR}")


# --------------------------------------------------
# 7. Search the stored vectors
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
# 8. Run example queries
# --------------------------------------------------

queries = [
    "How can I regain access to my account?",
    "What should I do if the application crashes?",
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


# --------------------------------------------------
# 9. Demonstrate persistence
# --------------------------------------------------

print("\n" + "=" * 65)
print("PERSISTENCE CHECK")
print("=" * 65)
print(f"Stored records: {collection.count()}")
print("Run this script again to test repeated ingestion.")
