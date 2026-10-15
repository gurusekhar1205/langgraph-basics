# Lesson 08: Generating Real Text Embeddings

## 1. Objective

In this experiment, we generated real text embeddings using the pretrained `all-MiniLM-L6-v2` model.

We learned how to:

- Load a pretrained embedding model.
- Convert sentences into numerical vectors.
- Inspect embedding dimensions.
- Normalize embedding vectors.
- Calculate cosine similarity using NumPy.
- Interpret actual similarity scores.

## 2. Model Used

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)
```

The model converts text into numerical representations that can be compared for semantic similarity.

For this experiment, each sentence produces a vector with 384 dimensions.

The model is pretrained, so we can generate embeddings without training it ourselves.

## 3. Generate Embeddings

```python
sentences = [
    "How can I reset my password?",
    "How do I recover my forgotten password?",
    "What is the weather today?",
    "I cannot log in because I forgot my credentials.",
]

embeddings = model.encode(sentences)
```

The `encode()` method generates one embedding for each input sentence.

Our output had this shape:

```text
(4, 384)
```

This means four sentences were embedded, with 384 numerical values per sentence.

The total number of numerical values is:

\[
4 \times 384 = 1536
\]

The shape is model-dependent. Another embedding model may return vectors of a different dimensionality.

## 4. Inspect an Embedding

```python
print(embeddings[0])
print(embeddings[0][:5])
print(len(embeddings[0]))
```

- `embeddings[0]` returns the vector for the first sentence.
- `embeddings[0][:5]` displays its first five values.
- `len(embeddings[0])` returns its dimensionality, which is 384 for this model.

The vector values are learned numerical representations. An individual dimension does not normally correspond directly to a human-readable concept.

## 5. Normalize the Embeddings

```python
normalized_embeddings = embeddings / np.linalg.norm(
    embeddings, axis=1, keepdims=True
)
```

The norm measures a vector's magnitude.

Normalization divides each vector by its magnitude, giving each nonzero vector a length of one.

For a vector \(v\), the normalized vector is:

\[
\hat{v} = \frac{v}{\|v\|}
\]

In our code:

- `axis=1` calculates a separate norm for each row.
- `keepdims=True` preserves a compatible array shape.
- NumPy broadcasting divides each vector by its own norm.

The normalized embedding matrix retains the shape `(4, 384)`.

## 6. Calculate Cosine Similarity

Cosine similarity compares the direction of two nonzero vectors.

\[
\operatorname{cosine}(A,B)
=
\frac{A\cdot B}{\|A\|\|B\|}
\]

Because our vectors are normalized, the cosine similarity is equal to their dot product.

```python
similarity_matrix = normalized_embeddings @ normalized_embeddings.T
```

Here:

- `@` performs matrix multiplication.
- `.T` transposes the embedding matrix.
- The result is a matrix containing every pairwise similarity score.

The dimensions are:

```text
(4, 384) × (384, 4) = (4, 4)
```

The resulting matrix contains 16 scores, including four diagonal scores comparing each sentence with itself.

## 7. Actual Results

The experiment produced these scores:

| Comparison | Cosine similarity |
|---|---:|
| Password reset vs forgotten password | 0.8247 |
| Password reset vs weather | 0.0438 |
| Password reset vs login credentials | 0.7055 |
| Forgotten password vs login credentials | 0.7029 |
| Forgotten password vs weather | 0.0531 |
| Weather vs login credentials | 0.0539 |

The diagonal entries were all `1.0000`.

### Interpretation

**Password reset vs forgotten password: 0.8247**

The two sentences express similar intentions using different wording.

**Password reset vs login credentials: 0.7055**

Both relate to account access, although one explicitly asks about resetting a password and the other describes being unable to log in.

**Password reset vs weather: 0.0438**

The two sentences concern different topics, and their embeddings have very low cosine similarity.

These results demonstrate that embeddings can represent semantic relationships beyond exact word matching.

## 8. Important Limitations

1. Similarity is not the same as correctness.
2. A score is not a percentage probability.
3. Similarity scores depend on the model, domain, and text.
4. A high similarity score does not guarantee that a retrieved chunk answers the question.
5. Embeddings do not replace the original text.
6. Real RAG systems need retrieval evaluation in addition to similarity calculations.

## 9. How This Fits into RAG

Our current experiment embeds four example sentences.

In the next stage, we will generate embeddings for our prepared document chunks.

The pipeline will be:

```text
Source documents
       |
       v
Chunking and metadata
       |
       v
Generate document embeddings
       |
       v
Store embeddings with chunk records
       |
       v
Embed a user query
       |
       v
Compare the query against document vectors
       |
       v
Retrieve relevant chunks
```

Document embeddings are generally generated during indexing. Query embeddings are generated when a user searches.

The query and document vectors must be compatible for meaningful comparison.

## 10. Key Takeaways

- `model.encode()` generates real embeddings from text.
- The selected model returns 384-dimensional vectors.
- NumPy can normalize vectors and calculate cosine similarity.
- Related sentences can have higher similarity even when their wording differs.
- Similarity scores are useful for ranking candidates, but they do not establish correctness.
- The next step is to apply embeddings to our own document chunks.