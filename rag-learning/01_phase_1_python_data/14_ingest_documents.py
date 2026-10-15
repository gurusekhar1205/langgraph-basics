
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "sample_docs"
DB_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "troubleshooting_documents"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# --------------------------------------------------
# 2. Load and chunk source documents
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

    for index, chunk_text in enumerate(split_texts):
        chunks.append({
            "id": f"{file_path.stem}_chunk_{index:03d}",
            "source": file_path.name,
            "chunk_index": index,
            "text": chunk_text,
        })

if not chunks:
    raise ValueError(f"No .txt documents found in {DATA_DIR}")

print(f"Documents found: {len(list(DATA_DIR.glob('*.txt')))}")
print(f"Chunks prepared: {len(chunks)}")


# --------------------------------------------------
# 3. Generate document embeddings
# --------------------------------------------------

model = SentenceTransformer(MODEL_NAME)

chunk_texts = [chunk["text"] for chunk in chunks]

embeddings = model.encode(
    chunk_texts,
    convert_to_numpy=True,
    normalize_embeddings=True,
)

print(f"Embedding matrix shape: {embeddings.shape}")


# --------------------------------------------------
# 4. Connect to the persistent ChromaDB collection
# --------------------------------------------------

client = chromadb.PersistentClient(path=str(DB_DIR))

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)

before_count = collection.count()

print(f"Records before ingestion: {before_count}")


# --------------------------------------------------
# 5. Store or update records
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

after_count = collection.count()

print(f"Records after ingestion: {after_count}")
print(f"Database location: {DB_DIR}")
print("Ingestion completed successfully.")
