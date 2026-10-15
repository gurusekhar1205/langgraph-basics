# Lesson 1: Document Basics for RAG

## 1. Learning objectives

After this lesson, you should understand:

- How to represent documents in Python.
- Why document IDs and metadata are useful.
- How to clean basic text.
- How to calculate simple text statistics.
- Why keyword matching is not semantic retrieval.

## 2. What is a document in RAG?

A document is a unit of information that a RAG application can process.

A document might be a troubleshooting guide, PDF manual, Markdown file, customer support article, database record, or section of a larger report.

For this exercise, we represent documents as Python dictionaries.

## 3. Understanding the document structure

Example:

```python
document = {
    "id": "doc_001",
    "text": "Users can reset a forgotten password.",
    "source": "account_guide.md",
    "category": "account",
}
```

Each field has a purpose:

| Field | Purpose |
|---|---|
| `id` | Identifies the document |
| `text` | Contains the document's actual content |
| `source` | Records where the information originated |
| `category` | Describes the document's subject |

A Python dictionary stores information as key-value pairs. For example, `document["text"]` returns the document text.

## 4. Why metadata matters

Metadata is information about a document rather than its main content.

Metadata can help a RAG system:

- Filter documents by category.
- Identify the original source.
- Display citations or source references.
- Track which documents were retrieved.
- Apply access-control rules where required.

Metadata does not automatically improve answer quality. The retrieval pipeline must use it appropriately.

## 5. Cleaning text

Real documents may contain extra spaces, blank lines, unusual formatting, or extraction artifacts.

```python
messy_text = "  Reset   your password.  "
cleaned_text = " ".join(messy_text.split())
```

`split()` separates text on whitespace and removes leading, trailing, and repeated whitespace. `" ".join(...)` joins the resulting words with a single space.

This is a simple example, not a universal cleaning strategy. Excessive normalization can damage code, tables, formatting, or other meaningful structure.

## 6. Text statistics

Python provides basic operations:

```python
text = "Reset your password"

character_count = len(text)
word_count = len(text.split())
```

- `len(text)` counts characters in the string.
- `len(text.split())` counts whitespace-separated pieces, a simple approximation of word count.

These statistics can help us inspect document sizes before chunking.

## 7. What is keyword search?

The exercise compares the words in a query with words in each document.

Query: `password reset`

Document: `Users can reset a forgotten password.`

Both query words appear in the document, so the document receives two matching words.

However, consider the query `locked out of my account`. It may relate to the same underlying problem, but its word overlap with the document is limited.

Keyword matching does not inherently understand synonyms, intent, or context.

This is a basic demonstration, not a production search engine, ranking model, or semantic retriever.

## 8. Where embeddings fit

An embedding model converts text into a numerical vector.

Conceptually:

`"reset my password"` -> `[0.12, -0.31, 0.48, ...]`

The numbers represent learned features of the text. They are not a direct dictionary definition of each word.

An embedding model can make semantically related text easier to retrieve even when the wording differs. Its performance depends on the model, data, and task.

We will study embeddings in Phase 2.

## 9. What is chunking?

Documents can be too long to process or retrieve as a single unit. Chunking divides a document into smaller pieces while attempting to preserve useful context.

For example:

- Full document: a 20-page troubleshooting manual.
- Chunks: smaller sections containing related troubleshooting instructions.
- Metadata: the source document and section information associated with each chunk.

Chunking strategy affects retrieval quality. We will study it after learning document preparation.

## 10. Important distinction

This lesson uses plain Python data structures and string operations.

It does not yet use an embedding model, vector database, retriever framework, LLM API, or complete RAG pipeline.

We are building the foundations first so each later component is understandable.

## 11. Practice questions

1. Why might a document need an ID?
2. What is the difference between document text and metadata?
3. What does `len(text)` return?
4. Why can keyword search miss a relevant document?
5. What problem does chunking solve?
6. Why should we preserve the original document source?

## 12. Completion checklist

- [ ] Run `01_document_basics.py`.
- [ ] Read and understand each section.
- [ ] Explain the purpose of all four document fields.
- [ ] Explain why keyword matching is limited.
- [ ] Answer the six practice questions.