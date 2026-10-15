# Lesson 4: Recursive Character Text Splitting

## 1. Learning objectives

- Understand recursive text splitting.
- Configure `RecursiveCharacterTextSplitter`.
- Learn how separator order affects chunk boundaries.
- Understand chunk size and overlap in LangChain.
- Preserve document metadata when creating chunks.
- Validate generated chunk lengths.

## 2. Why use a recursive splitter?

A simple character-based algorithm cuts text at fixed character positions. This can split words and sentences unnecessarily.

A recursive splitter tries to use more meaningful boundaries first, falling back to smaller boundaries when needed.

This often produces more readable chunks, although it cannot guarantee that every chunk is a complete sentence.

## 3. Installing and importing the package

The class is provided by `langchain-text-splitters`.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
```

This is a separate package from the main `langchain` package.

## 4. Configuring the splitter

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=80,
    chunk_overlap=20,
    length_function=len,
    separators=["\n\n", "\n", " ", ""],
)
```

- `chunk_size` sets the target maximum chunk length.
- `chunk_overlap` sets the target overlap between neighboring chunks.
- `length_function` determines how length is measured.
- `separators` controls which boundaries are preferred.

In this exercise, length is measured in characters, not model tokens.

## 5. Understanding separator order

The configured order is:

1. `"\n\n"`: paragraph boundaries.
2. `"\n"`: line boundaries.
3. `" "`: spaces between words.
4. `""`: individual characters as a final fallback.

The splitter tries to keep larger units together whenever possible, but the final result depends on the input and configuration.

## 6. Splitting text

```python
text_chunks = splitter.split_text(document["text"])
```

This returns a list of strings. Each string represents a chunk.

The returned strings do not automatically include our custom metadata, so we construct a dictionary for each chunk.

## 7. Preserving metadata

```python
chunk_record = {
    "chunk_id": "account_chunk_000",
    "document_id": "account",
    "source": "account.txt",
    "chunk_index": 0,
    "text": text_chunk,
}
```

This allows a retrieved chunk to be traced back to its source document.

In a real application, identifiers and metadata should be designed to remain reliable when documents are updated or reprocessed.

## 8. Understanding the validation

```python
assert all(
    len(chunk["text"]) <= CHUNK_SIZE
    for chunk in all_chunks
)
```

This checks that no generated chunk exceeds our configured size.

The check works because we use `length_function=len`. If a different length function were configured, validation should use the same definition of length.

## 9. Comparison with fixed character slicing

| Fixed character slicing | Recursive splitting |
|---|---|
| Cuts at calculated positions | Prefers configured separators |
| Can split words in the middle | Tries to preserve words and larger text units |
| Straightforward algorithm | More flexible boundary selection |
| Useful for understanding the basics | Useful for many practical text-preparation pipelines |

Neither strategy is automatically best for every dataset. Evaluate retrieval quality using representative questions.

## 10. Connection to RAG

The preparation pipeline now includes:

1. Loading documents.
2. Preserving source metadata.
3. Splitting documents into chunks.
4. Preparing chunks for embedding.

The next major topic is embeddings: representing text as numerical vectors so we can search for semantically related information.

## 11. Practice questions

1. Why can recursive splitting be preferable to fixed character slicing?
2. Why does separator order matter?
3. What does `length_function=len` mean?
4. Does recursive splitting guarantee complete sentences?
5. Why do we attach metadata after calling `split_text()`?
6. Why should we evaluate retrieval quality rather than choose a chunking strategy based only on appearance?

## 12. Completion checklist

- [ ] Run `04_recursive_chunking.py`.
- [ ] Inspect the chunk boundaries.
- [ ] Verify chunk lengths.
- [ ] Confirm source metadata is preserved.
- [ ] Answer the six practice questions.