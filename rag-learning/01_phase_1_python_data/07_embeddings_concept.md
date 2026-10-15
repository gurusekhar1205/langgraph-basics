# Lesson 07: Understanding Embeddings in RAG

## 1. Objective

The objective of this lesson is to understand embeddings before generating real embeddings with an AI model.

We will learn:

- What embeddings are.
- Why text is represented as numbers.
- What a vector and a dimension mean.
- How embeddings help semantic search.
- How embeddings fit into a RAG pipeline.
- The difference between illustrative vectors and real model-generated embeddings.

## 2. What Is an Embedding?

An embedding is a numerical representation of an input, such as text, that captures useful patterns in its meaning.

For example, consider these sentences:

```text
Sentence A: How can I reset my password?
Sentence B: How do I recover my forgotten password?
Sentence C: What is the weather today?
```

Sentence A and Sentence B express similar intentions.

Sentence C discusses a different subject.

An embedding model converts each sentence into a vector of numbers. A retrieval system can compare these vectors to estimate which texts are semantically similar.

The model does not simply assign one fixed number to each word. It generates a vector based on the input and the model's learned representations.

## 3. Why Do We Need Numbers?

Computers can process and compare numerical representations efficiently.

A text embedding might look conceptually like this:

```text
"How can I reset my password?"
              |
              v
       Embedding model
              |
              v
[0.12, -0.45, 0.78, 0.21, ...]
```

The numbers shown above are illustrative only. They are not actual embeddings from a trained model.

A real embedding vector may contain hundreds or thousands of dimensions, depending on the model.

Each dimension is a component of the learned numerical representation. It usually does not correspond directly to a human-readable feature such as "password" or "billing."

We generally interpret the vector as a whole rather than assigning an individual meaning to every dimension.

## 4. What Is a Vector?

A vector is an ordered collection of numerical values.

For example:

```python
vector = [0.12, -0.45, 0.78, 0.21]
```

This vector contains four values.

Its dimensionality is four.

In Python:

```python
print(len(vector))
```

Output:

```text
4
```

A real embedding model might produce a vector with 384 dimensions, 768 dimensions, or another model-specific size.

The dimensionality depends on the selected model. A larger dimension does not automatically mean that a model is better.

## 5. Understanding Dimensions

Suppose an embedding model produces 384 numbers for every input text.

Then:

```text
Input sentence
      |
      v
Embedding model
      |
      v
[384 numerical values]
```

The resulting vector has 384 dimensions.

If we embed 100 sentences using the same model, and each embedding has 384 dimensions, the embeddings can be arranged into a matrix of shape:

```text
100 x 384
```

Here:

- 100 represents the number of sentences.
- 384 represents the number of dimensions per sentence.

The matrix allows the retrieval system to compare a query embedding against document embeddings.

## 6. How Can Embeddings Capture Similarity?

Consider three example sentences:

```text
A: How can I reset my password?
B: How do I recover my forgotten password?
C: What is the weather today?
```

A well-suited embedding model should generally represent A and B as more similar than A and C.

Conceptually:

```text
A <------ similar meaning ------> B

A <------ different meaning ----> C
```

The system can measure similarity between the corresponding vectors.

However, embedding similarity is not a perfect understanding of meaning. Results depend on the model, the domain, the text, and the similarity metric.

## 7. What Is Cosine Similarity?

Cosine similarity is a common measure used to compare two vectors.

It measures the cosine of the angle between the vectors.

The formula is:

\[
\operatorname{cosine\ similarity}(A,B)
=
\frac{A \cdot B}{\|A\|\|B\|}
\]

Where:

- \(A\) and \(B\) are vectors.
- \(A \cdot B\) is their dot product.
- \(\|A\|\) and \(\|B\|\) are their magnitudes.

For nonzero vectors, cosine similarity ranges from -1 to 1 mathematically.

A value closer to 1 indicates that the vectors point in similar directions. A value around 0 indicates little directional similarity, and a negative value indicates opposing directions.

In practice, score distributions depend on the embedding model and data. A score of 0.8 should not automatically be interpreted as "80% meaning similarity."

Cosine similarity is one option; other retrieval systems may use dot product or Euclidean distance.

## 8. How Embeddings Work in RAG

A typical embedding-based RAG pipeline has two main stages.

### Stage A: Indexing documents

This usually happens before a user asks a question.

```text
Source documents
       |
       v
Split into chunks
       |
       v
Generate an embedding for each chunk
       |
       v
Store vectors with chunk text and metadata
```

Each chunk receives a numerical vector.

The system stores the vector along with the corresponding text and metadata, or stores references that allow the text and metadata to be retrieved.

### Stage B: Retrieving information

This happens when a user asks a question.

```text
User question
       |
       v
Generate query embedding
       |
       v
Compare with document vectors
       |
       v
Rank the most relevant chunks
       |
       v
Return retrieved text and metadata
       |
       v
Provide context to the language model
```

The query and document embeddings must be compatible. Typically, they are generated using the same embedding model and compatible settings.

The language model then uses the retrieved context to help answer the user's question.

## 9. Embeddings Versus Keyword Search

| Keyword search | Embedding-based search |
|---|---|
| Looks for matching terms or text patterns. | Compares numerical representations. |
| Can work well for exact error codes and names. | Can help find semantically related wording. |
| May miss paraphrases without matching terms. | May retrieve relevant paraphrases even when wording differs. |
| Often simpler to implement. | Requires an embedding model and vector comparison or search infrastructure. |

Neither approach is universally best.

For troubleshooting documents, exact error codes can be extremely important. A hybrid system may combine keyword retrieval with vector retrieval to benefit from both.

## 10. Important Distinctions

### An embedding is not the original text

The vector represents the text numerically. It does not replace the source text that the language model needs to read.

### An embedding is not a generated answer

Embeddings help retrieve relevant information. The language model usually generates the final answer using the retrieved context.

### An embedding is not a database

The vector is the numerical representation. A vector index or vector database stores or indexes vectors and supports similarity search.

### An embedding does not guarantee correctness

A semantically similar chunk may still be incorrect, outdated, or insufficient to answer the question. Retrieval quality must be evaluated.

## 11. Understanding the Python Experiment

The accompanying Python file contains manually created vectors:

```python
illustrative_vectors = [
    [0.90, 0.80, 0.10],
    [0.88, 0.82, 0.12],
    [0.10, 0.15, 0.95],
]
```

These vectors demonstrate the data structure only.

They are not produced by a trained model, so we must not use them to draw conclusions about actual sentence similarity.

The loop uses `zip()` to pair each sentence with its example vector:

```python
for text, vector in zip(texts, illustrative_vectors):
    print("Text:", text)
    print("Vector:", vector)
```

The following statement counts the dimensions:

```python
len(vector)
```

Because every illustrative vector contains three values, the output reports three dimensions.

In the next practical lesson, we will replace these invented numbers with embeddings generated by an actual model.

## 12. Key Takeaways

- An embedding is a numerical representation of input data.
- A vector is an ordered list of numerical values.
- The number of values determines the vector's dimensionality.
- Embeddings can help identify semantically related text.
- Cosine similarity is one common way to compare vectors.
- RAG typically embeds document chunks during indexing and user queries during retrieval.
- Retrieved text and metadata remain necessary to provide useful context to the language model.
- Real embeddings must be generated by an actual embedding model; manually invented vectors are only for demonstration.

## 13. Practice Questions

1. What is an embedding?
2. What is the difference between a vector and its dimensionality?
3. If a model returns a vector of 768 numbers, how many dimensions does it have?
4. Why can embedding search find relevant wording that keyword search misses?
5. What is cosine similarity used for?
6. Why do we embed the user's query?
7. Does an embedding directly generate the final RAG answer?
8. Why should we retain the original chunk text and metadata?
9. Why must query and document embeddings be compatible?
10. Why might a hybrid search system be useful for troubleshooting documents?

**Next lesson:** Generate real text embeddings in Python using a pretrained embedding model and inspect their dimensions and similarity scores.