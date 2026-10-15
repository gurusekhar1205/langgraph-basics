# Lesson 09: Generate Embeddings for Document Chunks

## 1. Objective

In this lesson, we applied a real embedding model to the chunks created from our sample documents.

We learned how to:

- Load the source documents.
- Split each document into chunks.
- Preserve the chunk metadata.
- Generate embeddings for multiple chunks in one call.
- Associate each vector with the correct chunk.
- Validate the dimensions and structure of the resulting dataset.

## 2. Why Embed Document Chunks?

In the previous lesson, we embedded four example sentences.

In a real RAG system, we need to embed the actual chunks that the system will search.

For example, a source document might contain:

```text id="apw6yr"
Users can reset a forgotten password from the login page.
```

After chunking, this text becomes a record with metadata.

The embedding model converts the chunk text into a numerical vector.

```text id="8yewf3"
Chunk text
    |
    v
Embedding model
    |
    v
384-dimensional vector
```

This vector can later be compared with the embedding of a user query.

## 3. Load and Split the Documents

The program loads the `.txt` files from `data/sample_docs`.

It then uses `RecursiveCharacterTextSplitter` to create chunks.

Each chunk contains:

- `chunk_id`: Unique identifier for the chunk.
- `document_id`: Identifier for the original document.
- `chunk_index`: Position of the chunk within its document.
- `source`: Original filename.
- `text`: Actual chunk content.
- `char_count`: Number of characters in the chunk.

This metadata is retained when embeddings are generated.

## 4. Extract the Text

```python id="lqlknm"
texts = [chunk["text"] for chunk in chunks]
```

This list comprehension extracts the text from each chunk record.

The embedding model needs the text as input. We keep the metadata in the original chunk records so we can reconnect each generated vector with its source.

## 5. Generate Embeddings in a Batch

```python id="83g4e3"
embeddings = model.encode(
    texts,
    normalize_embeddings=True,
    show_progress_bar=True,
)
```

The `encode()` method processes the list of chunk texts and returns an embedding for each input.

We use two options:

- `normalize_embeddings=True`: Requests unit-normalized embedding vectors.
- `show_progress_bar=True`: Displays a progress indicator.

Batch processing allows us to submit multiple texts in one call rather than manually invoking the model for every chunk.

For this model, six chunks produce an embedding matrix with shape:

```text id="08j7aj"
(6, 384)
```

This means six vectors, with 384 numerical values per vector.

The total number of embedding values is:

\[
6 \times 384 = 2304
\]

The actual number of chunks depends on the source text and splitting configuration.

## 6. Associate Embeddings with Chunk Records

```python id="u4g7tg"
for chunk, embedding in zip(chunks, embeddings):
    chunk["embedding"] = embedding
```

The `zip()` function pairs each chunk record with the corresponding embedding.

The vector is attached to the same record as its text and metadata.

Conceptually, a record now contains:

```python id="54q7hl"
{
    "chunk_id": "account_chunk_000",
    "document_id": "account",
    "source": "account.txt",
    "text": "Example chunk text...",
    "char_count": 50,
    "embedding": "384 numerical values"
}
```

The text above is illustrative; the actual embedding is a NumPy array rather than a string.

Maintaining this association is essential. When a vector is retrieved during search, the application must be able to identify the text and source that belong to it.

## 7. Why Normalize the Embeddings?

The program requests normalized embeddings from the model:

```python id="p38b8f"
normalize_embeddings=True
```

A normalized nonzero vector has a magnitude of approximately one.

We inspect its magnitude with:

```python id="a1dm8y"
np.linalg.norm(embedding)
```

This should be approximately `1.0`.

Normalized embeddings are convenient for cosine-similarity calculations because the dot product of two unit vectors equals their cosine similarity.

Normalization is not mandatory for every retrieval system. The correct choice depends on the similarity metric and model.

## 8. Validate the Dataset

The program performs several checks.

### Number of chunks and embeddings

```python id="72kbrb"
assert len(chunks) == len(embeddings)
```

Every chunk must have a corresponding embedding.

### Embedding dimensionality

```python id="45ax4m"
assert all(len(chunk["embedding"]) == 384 for chunk in chunks)
```

Every vector must contain 384 values for this model.

### Unique chunk IDs

```python id="6ds6fj"
assert len({chunk["chunk_id"] for chunk in chunks}) == len(chunks)
```

This catches duplicate IDs.

### Required content

```python id="f0h2r5"
assert all(chunk["source"] and chunk["text"] for chunk in chunks)
```

This checks that each record has a source and non-empty text.

These checks catch basic data preparation errors before the vectors are used for retrieval.

## 9. Are the Embeddings Stored Permanently?

No. In this experiment, the embeddings exist in Python memory while the program is running.

When the process exits, these in-memory records are lost unless we save them or send them to a storage system.

A production-style RAG pipeline typically stores:

- The embedding vector in a vector index or vector database.
- The corresponding chunk text.
- Metadata such as the source filename and document ID.

The storage system may keep these fields together or use IDs to link the vector index to a separate document store.

The next stages of this project will introduce vector search and persistent storage.

## 10. Current RAG Pipeline

We have now completed the following stages:

```text id="xdj5p2"
Source documents
       |
       v
Load and clean text
       |
       v
Split documents into chunks
       |
       v
Attach metadata
       |
       v
Generate embeddings
       |
       v
Associate vectors with chunk records
```

The next stage is to search those vectors using a query embedding.

For example:

```text id="xk78un"
Query:
"How do I regain access to my account?"
```

We will generate an embedding for the query and compare it with the stored document embeddings.

The highest-ranked chunks become candidates for retrieval. We must still evaluate whether those chunks actually answer the question.

## 11. Key Takeaways

- We should embed the chunks our RAG system will actually search.
- Batch embedding is convenient for processing multiple chunks.
- Metadata must remain associated with the corresponding vector.
- The model used here generates 384-dimensional embeddings.
- Normalization makes cosine-similarity calculations convenient.
- In-memory embeddings are not persistent storage.
- Retrieval requires comparing query embeddings with document embeddings.

**Next lesson:** Learn how to calculate query-to-document similarity and retrieve the most relevant chunks from our own dataset.