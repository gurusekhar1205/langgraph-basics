
# Lesson 14: Separating Ingestion and Retrieval

## 1. Objective

In Lesson 13, one program read documents, split them into chunks, generated document embeddings, stored them in ChromaDB, and searched for user queries.

That program worked, but it repeated the document preparation and embedding steps every time it ran.

In this lesson, we separate those responsibilities into two programs:

- 14_ingest_documents.py
- 14_search_documents.py

The goal is to avoid re-embedding the complete document collection during normal query processing.

## 2. What is ingestion?

Ingestion prepares documents for retrieval.

The ingestion workflow is:

1. Read source files.
2. Clean document text.
3. Split documents into chunks.
4. Assign IDs and metadata.
5. Generate embeddings for the chunks.
6. Store chunk text, vectors, and metadata in ChromaDB.

The ingestion program is normally run when documents are first added or updated.

## 3. What is retrieval?

Retrieval finds stored chunks relevant to a user query.

The retrieval workflow is:

1. Receive a user query.
2. Generate the query embedding.
3. Open the existing ChromaDB collection.
4. Search the stored vectors.
5. Return the top K matching chunks and their metadata.

Retrieval does not need to read and chunk the original documents.

## 4. Why separate these operations?

Suppose a knowledge base contains 100,000 chunks.

If every query regenerates all 100,000 document embeddings, the application repeats expensive work unnecessarily.

With separated ingestion and retrieval:
- Document embeddings are generated during ingestion.
- Stored vectors are reused across queries.
- Each query requires a query embedding and a vector search.
- Document changes can be handled through a controlled ingestion process.

This separation improves efficiency and makes the application easier to maintain.

## 5. File responsibilities

### 14_ingest_documents.py

This file:
- Reads the sample text files.
- Splits text into chunks.
- Generates document embeddings.
- Connects to persistent ChromaDB storage.
- Uses upsert to insert or update records.

### 14_search_documents.py

This file:
- Opens the existing ChromaDB collection.
- Loads the same embedding model.
- Embeds each user query.
- Searches stored vectors.
- Returns text, IDs, metadata, and distances.

It does not read the original text files or regenerate document embeddings.

## 6. Why use the same collection name and path?

Both programs use:

Collection name:
troubleshooting_documents

Database directory:
chroma_db

The directory and collection name must match. Otherwise, the retrieval program may open a different database or fail to find the collection.

## 7. Why use the same embedding model?

Document and query vectors must be produced in a compatible vector space.

This project uses:

sentence-transformers/all-MiniLM-L6-v2

The model generates 384-dimensional embeddings.

If we change the model, we generally need to re-embed the documents and rebuild or migrate the collection appropriately.

The database does not automatically guarantee that the selected query model matches the model used during ingestion.

## 8. What does upsert do?

Upsert inserts records or updates existing records with matching IDs.

Stable IDs let us rerun ingestion without duplicating the same IDs.

However, upsert does not automatically delete obsolete IDs. If documents are removed or chunk IDs change, the ingestion pipeline must identify and remove stale records.

## 9. Execution order

First, run ingestion:

python 14_ingest_documents.py

Then, run retrieval:

python 14_search_documents.py

You can run the retrieval program repeatedly without rerunning ingestion, provided the stored collection already exists and is populated.

## 10. What should happen?

The ingestion program should report six prepared chunks and six stored records.

The retrieval program should report six stored records and return three ranked chunks for each of its example queries.

The ranking should be consistent with the previous lesson when the source data, model, preprocessing, and collection remain unchanged.

## 11. Important distinction

Separating the programs does not mean that no model is loaded during retrieval.

The retrieval program still loads the embedding model because it must convert each query into a vector.

What it avoids is regenerating embeddings for every document chunk.

## 12. Production considerations

A production ingestion pipeline should also handle:

- Detecting changed source documents.
- Removing obsolete chunks.
- Stable IDs and duplicate prevention.
- Embedding-model version tracking.
- Logging and error handling.
- Rebuilding indexes when the embedding model changes.

These are beyond the scope of the first separated implementation.

## 13. Exercises

1. Run ingestion once and record the before and after counts.
2. Run the retrieval program three times without rerunning ingestion.
3. Verify that retrieval still works.
4. Change a query to ask about billing.
5. Explain why the retrieval program still loads the embedding model.
6. Explain what happens when a source document changes.
7. Explain why upsert alone cannot remove stale records.

## 14. Key takeaways

- Ingestion prepares and stores document chunks.
- Retrieval searches the existing stored chunks.
- Document embeddings do not need to be regenerated for every query.
- Query embeddings are still required.
- Both programs must use the same database path and collection.
- The same embedding model must be used for compatible document and query vectors.
- Robust document updates require more than upsert alone.
