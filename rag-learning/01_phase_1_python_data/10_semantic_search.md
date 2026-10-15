# Lesson 10: Semantic Search Using Embeddings

## 1. Objective

In this lesson, we build our first semantic search program using real document embeddings.

We will learn how to:

- Convert a user's question into an embedding.
- Compare a query embedding against document embeddings.
- Calculate cosine similarity scores.
- Rank chunks from most similar to least similar.
- Retrieve the top-k results.
- Return the relevant text and source metadata.

## 2. What Is Semantic Search?

Semantic search retrieves information based on the meaning represented by text rather than relying exclusively on exact word matches.

Consider this user question:

```text
How can I regain access to my account?
```

Our account document contains:

```text
Users can reset a forgotten password from the login page.
```

The wording differs, but both texts concern account access.

An embedding model can represent these texts in a way that allows a similarity measure to identify their relationship.

Semantic search uses this similarity to rank candidate documents or chunks.

## 3. Keyword Search Versus Semantic Search

Suppose a document says:

```text
Users can reset a forgotten password from the login page.
```

A keyword search for `forgotten password` may find the document because the exact words appear in the text.

However, a query such as `How do I regain account access?` may not share those exact words.

Semantic search compares embeddings and can retrieve related content despite different wording.

| Keyword search | Semantic search |
|---|---|
| Matches terms or text patterns. | Compares numerical representations of text. |
| Often useful for exact error codes and names. | Can find paraphrases and related concepts. |
| May miss relevant wording without matching terms. | May retrieve related but not fully relevant content. |

Neither method is perfect. Hybrid search can combine keyword and vector-based retrieval.

## 4. The Semantic Search Pipeline

Our program follows this sequence:

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
Preserve metadata
       |
       v
Generate document embeddings
       |
       v
Receive user query
       |
       v
Generate query embedding
       |
       v
Calculate similarity scores
       |
       v
Rank the chunks
       |
       v
Return the top-k results
```

The document embeddings are generated before searching. Each new query is embedded when the search function runs.

In this learning program, the document embeddings are regenerated every time the script starts. A larger application would typically persist the document vectors or store them in a vector database.

## 5. Generate Document Embeddings

```python
document_texts = [chunk["text"] for chunk in chunks]

document_embeddings = model.encode(
    document_texts,
    normalize_embeddings=True,
    show_progress_bar=True,
)
```

The list comprehension extracts the text from every chunk.

The model then converts each text into a numerical vector.

With six chunks and the selected 384-dimensional model, the expected matrix shape is:

```text
(6, 384)
```

The matrix contains six vectors, each with 384 values.

Normalization ensures that each nonzero embedding has a magnitude of approximately one.

## 6. Generate the Query Embedding

```python
query_embedding = model.encode(
    query,
    normalize_embeddings=True,
)
```

This converts the user's question into an embedding.

For example:

```text
Query:
How can I regain access to my account?
```

The resulting query vector has 384 dimensions.

The query and document embeddings must be compatible. Here, both are produced by the same model with the same normalization setting.

## 7. Calculate Similarity Scores

```python
scores = document_embeddings @ query_embedding
```

Both the query vector and the document vectors are normalized.

For normalized vectors, the dot product equals cosine similarity:

\[
\operatorname{cosine}(A,B)=A\cdot B
\]

The `@` operator performs matrix multiplication.

The output contains one similarity score for each document chunk.

With six chunks, there are six scores.

A higher score indicates greater directional similarity between the query embedding and the corresponding chunk embedding.

A score is not a probability, a percentage of correctness, or a guarantee that the chunk answers the question.

## 8. Rank the Results

```python
ranked_indices = np.argsort(scores)[::-1]
```

`np.argsort(scores)` returns indices in ascending score order.

The slice `[::-1]` reverses that order so the highest score appears first.

For example:

```python
scores = np.array([0.20, 0.85, 0.40])
```

Ascending order:

```text
Indices: [0, 2, 1]
```

After reversing:

```text
Indices: [1, 2, 0]
```

The corresponding scores are now ordered from highest to lowest.

These are indices into the chunk list, so the program can retrieve the matching text and metadata.

## 9. What Does Top-k Mean?

```python
TOP_K = 3
```

Top-k retrieval returns the k highest-ranked chunks.

If `top_k=3`, the program returns at most three results.

This does not mean the program has proved that three chunks are relevant. It simply selects the three highest-scoring candidates.

The best value of k depends on the documents, questions, and downstream language model. We should evaluate retrieval quality rather than automatically assuming that three is always correct.

## 10. Why Return Metadata?

The program returns:

```python
{
    "chunk_id": "...",
    "source": "...",
    "text": "...",
    "score": 0.0
}
```

The fields serve different purposes:

- `chunk_id` identifies the chunk.
- `source` identifies the source file.
- `text` provides the retrieved information.
- `score` helps rank the candidates.

Metadata makes it possible to trace retrieved content to its original document.

This is particularly useful for troubleshooting agents that need to identify which internal guide or runbook contains the relevant information.

## 11. Understanding the Search Function

The function:

```python
def semantic_search(query, top_k=3):
```

accepts a user query and the maximum number of results to return.

It then:

1. Validates the query.
2. Embeds the query.
3. Calculates similarity against every document embedding.
4. Sorts the scores.
5. Selects the top-ranked indices.
6. Builds result dictionaries with the text and metadata.
7. Returns the results.

The function raises a `ValueError` for an empty query or an invalid top-k value.

This is a simple in-memory implementation designed for learning.

## 12. What Happens After Retrieval?

Our program currently stops after returning ranked chunks.

A complete RAG pipeline continues:

```text
User question
      |
      v
Semantic retrieval
      |
      v
Top-ranked chunks
      |
      v
Construct a context for the language model
      |
      v
Generate an answer
      |
      v
Return the answer with supporting sources
```

Retrieval and answer generation are separate operations.

Semantic search finds candidate evidence. The language model uses the retrieved evidence to generate an answer.

The quality of the final answer depends on retrieval quality, the relevance and completeness of the context, and the model's behavior.

## 13. Limitations of This Implementation

1. The document embeddings exist only in memory.
2. The program recalculates embeddings each time it starts.
3. Every query is compared against every document vector.
4. Similarity alone does not guarantee that a result answers the question.
5. The fixed top-k value may return irrelevant chunks when no good match exists.
6. The current program does not generate an answer with an LLM.
7. The example uses a small dataset and is not a production vector database.

These limitations are intentional. This implementation helps us understand retrieval before introducing more infrastructure.

## 14. Key Takeaways

- Semantic search uses embeddings to find related text.
- The user's query receives its own embedding.
- Query and document embeddings must be compatible.
- Cosine similarity helps rank candidate chunks.
- Top-k retrieval selects a fixed number of highest-scoring candidates.
- Source metadata supports traceability.
- Retrieval is one stage of RAG, not the complete RAG pipeline.
- Search quality must be evaluated with realistic queries.

## 15. Practice Questions

1. Why do we generate a query embedding?
2. Why do we embed document chunks before searching?
3. What does the `@` operator do in this program?
4. Why do we reverse the output of `np.argsort()`?
5. What does `top_k=3` mean?
6. Why do we return the source filename?
7. Does the highest similarity score guarantee a correct answer?
8. What is the difference between semantic search and a complete RAG pipeline?
9. Why is recalculating all document embeddings on every run inefficient for a large dataset?
10. How could we combine semantic search with exact keyword matching?

**Next lesson:** Learn how to evaluate retrieved results, then explore vector indexes and persistent vector storage.