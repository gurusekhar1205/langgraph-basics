# Lesson 06: Preparing Document Chunks with Metadata

## 1. Objective

In this lesson, we prepare document chunks for the next stages of a Retrieval-Augmented Generation (RAG) pipeline.

We will learn how to:

- Load text documents.
- Split documents into chunks.
- Assign a unique ID to every chunk.
- Preserve the identity of the original document.
- Store useful metadata alongside the chunk text.
- Validate the prepared chunks before generating embeddings.

## 2. What Is Metadata?

Metadata is information that describes a piece of data.

In a RAG system, the actual chunk contains the document content, while metadata describes the chunk.

For example:

```python
chunk = {
    "chunk_id": "billing_chunk_001",
    "document_id": "billing",
    "chunk_index": 1,
    "source": "billing.txt",
    "text": "Contact the billing team for payment disputes.",
    "char_count": 46,
}
```

Here:

- `chunk_id` uniquely identifies this chunk.
- `document_id` identifies the original document.
- `chunk_index` indicates the chunk's position within that document.
- `source` identifies the source filename.
- `text` contains the actual content.
- `char_count` records the number of characters in the chunk.

The first five fields have different purposes. In particular, `chunk_id` identifies an individual chunk, while `document_id` identifies the original document.

## 3. Why Is Metadata Important in RAG?

Imagine a troubleshooting agent searches four internal documents and finds a solution.

Without source metadata, the agent might retrieve the solution text but have no convenient way to identify which file it came from.

With metadata, the system can retain the source alongside the retrieved content.

Metadata can support:

1. **Source identification:** Identify the file that contains a retrieved answer.
2. **Filtering:** Restrict retrieval to a category, department, document, or other supported field.
3. **Traceability:** Investigate which documents contributed to an answer.
4. **Organization:** Group chunks belonging to the same document.
5. **Evaluation:** Analyze which sources are retrieved for particular test questions.

Metadata does not itself make a chunk semantically relevant. Retrieval still needs an appropriate search method.

## 4. Why Do We Need a Unique Chunk ID?

Consider two documents:

```text
account.txt
billing.txt
```

Both documents might contain a chunk with index `0`.

Therefore, the chunk index alone cannot uniquely identify a chunk across the entire dataset.

We construct the ID using the original document name and chunk index:

```python
chunk_id = f"{document['document_id']}_chunk_{chunk_index:03d}"
```

For example:

```text
account_chunk_000
account_chunk_001
billing_chunk_000
billing_chunk_001
```

The format `:03d` displays an integer using at least three digits, adding leading zeros when needed.

For example:

```python
f"{0:03d}"  # "000"
f"{1:03d}"  # "001"
f"{12:03d}" # "012"
```

This makes IDs easier to read and sort.

The ID must be unique across the prepared dataset. If multiple files can have the same stem, such as two different `guide.txt` files in different folders, we would need to include additional path information to avoid collisions.

## 5. Understanding the Code

### Step 1: Load documents

```python
for file_path in sorted(DATA_DIR.glob("*.txt")):
    text = file_path.read_text(encoding="utf-8")
```

`glob("*.txt")` selects text files in the sample directory.

`sorted()` makes the file-processing order predictable.

`read_text()` reads the file contents into a Python string.

### Step 2: Clean the text

```python
text = " ".join(text.split())
```

This normalizes whitespace by replacing sequences of whitespace characters with a single space and removing leading and trailing whitespace.

For example:

```text
Before:
Users can reset a forgotten password
from the login page.

After:
Users can reset a forgotten password from the login page.
```

This simple cleaning step is useful for these sample documents, but it is not appropriate for every dataset. In code, tables, or structured documents, newlines and spacing can carry important meaning.

### Step 3: Split the document

```python
text_chunks = splitter.split_text(document["text"])
```

The splitter returns a list of strings.

At this point, each string contains the chunk text, but the list does not automatically include our custom metadata fields.

### Step 4: Attach metadata

```python
chunk_record = {
    "chunk_id": f"{document['document_id']}_chunk_{chunk_index:03d}",
    "document_id": document["document_id"],
    "chunk_index": chunk_index,
    "source": document["source"],
    "text": chunk_text,
    "char_count": len(chunk_text),
}
```

We combine the chunk text with the fields that describe it.

This creates a structured Python dictionary for every chunk.

### Step 5: Add the chunk to the collection

```python
chunks.append(chunk_record)
```

The `chunks` list becomes our collection of prepared records.

Conceptually:

```text
chunks
  |
  +-- account_chunk_000
  |      +-- text
  |      +-- source
  |      +-- document_id
  |
  +-- account_chunk_001
  |      +-- text
  |      +-- source
  |      +-- document_id
  |
  +-- billing_chunk_000
         +-- text
         +-- source
         +-- document_id
```

## 6. Why Do We Validate the Chunks?

Preparing data is not complete until we check that it meets our expectations.

Our code performs four checks.

### Check 1: Unique IDs

```python
assert len(chunk_ids) == len(set(chunk_ids))
```

A Python set contains unique values. If the list contains duplicate chunk IDs, its length will be greater than the set's length.

Unique IDs help prevent ambiguity when tracking or storing chunks.

### Check 2: Non-empty text

```python
assert all(chunk["text"].strip() for chunk in chunks)
```

This verifies that every chunk contains non-whitespace text.

Empty chunks waste processing and can create meaningless records.

### Check 3: Maximum chunk size

```python
assert all(chunk["char_count"] <= CHUNK_SIZE for chunk in chunks)
```

This confirms that no chunk exceeds our configured 80-character limit.

This assertion is valid because our configured length function is `len`, which counts characters.

### Check 4: Required metadata

```python
assert all(
    chunk["document_id"] and chunk["source"]
    for chunk in chunks
)
```

This verifies that every chunk has a document ID and source filename.

These assertions raise an `AssertionError` if a condition fails. For a small learning project, this is convenient. In production systems, use explicit validation and error handling appropriate to the application.

## 7. How Does This Connect to Embeddings?

We now have structured chunks that can be processed individually.

The next stage of a typical RAG pipeline is embedding generation.

```text
Source documents
       |
       v
Load and clean text
       |
       v
Split into chunks
       |
       v
Attach metadata
       |
       v
Validate chunk records
       |
       v
Generate an embedding for each chunk
       |
       v
Store embedding + text + metadata
       |
       v
Search for relevant chunks
```

An embedding is a numerical representation of text. It helps a retrieval system identify content that is semantically similar to a query.

The metadata is not normally concatenated into the chunk text automatically. We decide separately which content to embed and which fields to retain as metadata.

For example, the embedding input could be:

```text
Contact the billing team for payment disputes.
```

The associated metadata could remain:

```python
{
    "source": "billing.txt",
    "document_id": "billing",
    "chunk_index": 1
}
```

When a chunk is retrieved, the system can return both its content and its metadata.

## 8. Key Takeaways

- Metadata describes a chunk; it is different from the chunk's actual text.
- A unique chunk ID helps identify an individual record.
- A document ID connects chunks to their original document.
- The chunk index indicates the chunk's position within its source document.
- Source metadata helps with traceability and filtering.
- Validation catches basic data-preparation errors before embedding generation.
- Text and metadata can be stored together while being used for different purposes.

## 9. Practice Questions

1. What is the difference between `chunk_id` and `document_id`?
2. Why is `chunk_index` useful?
3. Why might storing only the chunk text be insufficient for a troubleshooting agent?
4. What does `file_path.stem` return?
5. Why do we check that chunk IDs are unique?
6. Does metadata replace embeddings for semantic search?
7. Should every metadata field be included in the text passed to the embedding model?
8. What should happen if a source document is empty?

**Next lesson:** Learn what embeddings are, how text becomes a numerical vector, and why semantic similarity can retrieve relevant information even when the query uses different words from the source document.