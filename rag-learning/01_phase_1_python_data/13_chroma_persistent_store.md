
# Lesson 13: Persistent Vector Storage with Chroma

## 1. Objective

In previous lessons, we generated embeddings and performed semantic search using NumPy.

However, the document embeddings existed only in the running Python process. The program had to reload the documents and regenerate the embeddings when it started again.

In this lesson, we use Chroma to store:
- Chunk IDs
- Original chunk text
- Embedding vectors
- Source filenames
- Chunk metadata

The goal is to reuse stored document vectors across program runs.

## 2. What is a vector database?

A vector database stores numerical vectors and provides similarity-search operations.

In a RAG application, each document chunk can have:
- A unique ID
- Text content
- An embedding vector
- Metadata describing its origin

When a user asks a question, the application embeds the query and searches for similar stored vectors.

The database returns the associated text and metadata for the most similar records.

## 3. What is Chroma?

Chroma is a vector database that can run locally or be used in other deployment configurations.

For this lesson, we use its local persistent client.

The database files are stored in:

rag-learning/01_phase_1_python_data/chroma_db/

The database directory is generated data, not source code. It should not normally be committed to Git.

## 4. Important terminology

### Client

The client provides access to the database.

Example:

client = chromadb.PersistentClient(path=str(DB_DIR))

The path identifies the local persistent storage directory.

### Collection

A collection groups related records.

Example:

collection = client.get_or_create_collection(
    name="troubleshooting_documents",
    metadata={"hnsw:space": "cosine"},
)

The collection is similar in purpose to a table in a relational database, although its search capabilities are designed for vectors.

### Record

A record contains a vector and associated information.

Example:

ID:
account_chunk_000

Text:
Users can reset a forgotten password from the login page.

Metadata:
source = account.txt
chunk_index = 0

Embedding:
A vector of 384 floating-point numbers.

### Upsert

Upsert inserts new records or updates existing records with matching IDs.

Example:

collection.upsert(
    ids=chunk_ids,
    documents=chunk_texts,
    embeddings=embedding_vectors,
    metadatas=metadata_records,
)

Stable IDs make repeated ingestion of the same chunks idempotent with respect to those IDs: the records are updated rather than duplicated.

Important:
Upsert does not automatically remove old IDs that are no longer present in the source documents. A production ingestion process must handle deleted or changed chunks.

### Persistence

Persistence means data survives after the program exits.

The database is saved locally rather than existing only in Python memory.

## 5. Workflow

1. Read the sample documents.
2. Clean and split the documents into chunks.
3. Assign a stable ID and metadata to each chunk.
4. Load the embedding model.
5. Generate normalized document embeddings.
6. Open the persistent Chroma database.
7. Create or retrieve the collection.
8. Upsert IDs, texts, vectors, and metadata.
9. Embed a user query.
10. Ask Chroma to retrieve the nearest stored vectors.
11. Read the associated text, metadata, and distance scores.

## 6. Understanding the embedding shape

The script prints an embedding matrix shape.

For six chunks and 384-dimensional embeddings, the shape is:

(6, 384)

This means:
- There are six document vectors.
- Each vector contains 384 numbers.

The database stores these vectors alongside their associated records.

## 7. Cosine similarity versus cosine distance

Earlier lessons used cosine similarity to rank document vectors.

Chroma is configured here to use cosine distance.

For normalized vectors:

Cosine distance = 1 - cosine similarity

Examples:

Similarity = 0.90
Distance = 0.10

Similarity = 0.70
Distance = 0.30

Similarity = 0.20
Distance = 0.80

Smaller cosine distance indicates greater similarity.

The script prints both distance and the corresponding similarity value for interpretation.

These similarity values are not probabilities or guaranteed relevance percentages.

## 8. Why do we normalize embeddings?

The script uses:

normalize_embeddings=True

Normalization scales each non-zero vector to unit length.

For normalized vectors, the dot product equals cosine similarity.

Chroma calculates the configured cosine distance during retrieval.

Both the document embeddings and query embeddings must use compatible preprocessing and the same embedding model.

## 9. Why use persistent storage?

Without persistent storage, the application must reconstruct its in-memory document vectors every time it starts.

With persistent Chroma storage:
- Stored document vectors can be reused.
- Text and metadata are stored with the records.
- New queries can search the existing collection.
- Repeated upserts with stable IDs do not create duplicate IDs.

The embedding model is still needed to embed new queries.

The current demonstration still reads, chunks, and embeds all source documents before upserting them. This is intentional for the first lesson.

In a later lesson, ingestion and query-time retrieval can be separated so that existing documents are not re-embedded on every application start.

## 10. What does the search function do?

The search function performs these steps:

1. Accept a user query.
2. Generate the query embedding.
3. Send that vector to Chroma.
4. Request the top K records.
5. Receive IDs, text, metadata, and distances.
6. Return the matching records in rank order.

The database does not generate the final answer. It retrieves information that can later be passed to an LLM.

## 11. Repeated execution

Run:

python 13_chroma_persistent_store.py

On the first run, the collection will normally contain zero records before upsert and six after upsert.

On a second run, the records-before-upsert count should normally be six, and the count after upsert should remain six.

This demonstrates that the records persist and that the same IDs are reused.

If the source files change, obsolete chunk IDs are not automatically deleted by upsert. The ingestion process must reconcile old and new records.

## 12. Git considerations

The chroma_db directory contains generated database data.

Keep it out of version control unless you have a deliberate reason to version a small database fixture.

The root .gitignore already excludes directories named chroma_db.

Commit the Python source and Markdown notes, not the local database files.

## 13. Limitations

1. The collection is local to this environment.
2. The current script still regenerates document embeddings during ingestion.
3. Stable IDs do not automatically remove obsolete records.
4. The dataset is tiny and does not demonstrate production performance.
5. Similarity is not the same as correctness.
6. A vector database does not replace retrieval evaluation.

## 14. Exercises

1. Run the script twice and compare the record counts.
2. Ask a billing-related question.
3. Compare the distance scores of relevant and irrelevant chunks.
4. Explain why query embeddings are still required.
5. Explain why upsert does not remove obsolete chunk IDs.
6. Explain why the database directory should generally be excluded from Git.

## 15. Key takeaways

- Chroma stores vectors and their associated text and metadata.
- A persistent client stores data on disk.
- Collections organize related records.
- Upsert inserts or updates records by ID.
- Cosine distance is lower for more similar vectors.
- Persistent storage avoids rebuilding the database from scratch at query time.
- Ingestion and retrieval are separate responsibilities in a well-designed RAG system.
