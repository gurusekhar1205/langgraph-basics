# Lesson 11: Evaluating Retrieval Quality

## 1. Objective

This lesson introduces a basic method for evaluating a semantic search system.

We will learn:

- Why retrieval needs evaluation.
- How to prepare test queries with expected sources.
- What Hit@K means.
- How to calculate a retrieval hit rate.
- Why a high retrieval score does not guarantee a correct final answer.

## 2. Why Evaluate Retrieval?

A semantic search system always ranks available chunks, but that does not mean the results are useful.

For example, a query about password recovery might return a billing document in its top results.

We need a systematic way to check whether relevant information is being retrieved.

Evaluation helps us compare changes to:

- Embedding models.
- Chunk sizes and overlap.
- Search methods.
- Retrieval parameters.
- Metadata filters.

Without evaluation, we might change the system and mistakenly assume it improved simply because the output looks reasonable.

## 3. Create Test Cases

Each test case contains a query and an expected source.

```python id="8j7r6c"
test_cases = [
    {
        "query": "How can I regain access to my account?",
        "expected_source": "account.txt",
    },
    {
        "query": "What should I do if the application crashes?",
        "expected_source": "technical.txt",
    },
    {
        "query": "How do I handle duplicate payments?",
        "expected_source": "billing.txt",
    },
]
```

These test cases are a small starting point.

For a serious evaluation, we should use more questions, multiple phrasings, ambiguous cases, and questions that cannot be answered from the available documents.

## 4. What Is Hit@K?

Hit@K measures whether at least one relevant result appears among the top K retrieved results.

For example, with `K=3`, a query counts as a hit if its expected source appears in any of the three results.

```python id="l2mgzy"
hit = expected_source in retrieved_sources
```

This expression evaluates to `True` when the expected source appears in the list, and `False` otherwise.

Hit@K is a simple retrieval metric. In this example, relevance is defined using the expected filename. More advanced evaluation can use relevance labels for individual chunks.

## 5. Calculate the Hit Rate

```python id="9e2eqe"
hit_rate = hits / len(test_cases)
```

If two out of three queries are hits:

\[
\operatorname{Hit@3}=\frac{2}{3}\approx66.67\%
\]

If all three are hits:

\[
\operatorname{Hit@3}=\frac{3}{3}=100\%
\]

This is a measurement of the test set, not a guarantee of performance on every possible query.

## 6. What Does a Hit Prove?

A hit proves only that the expected source appeared within the selected top K results according to our evaluation rule.

It does not prove that:

- The exact best chunk was retrieved.
- Every returned chunk was relevant.
- The retrieved text contains enough context.
- The final language-model answer will be correct.
- The system will perform equally well on unseen queries.

For example, a query may retrieve the correct file but return a chunk that contains only part of the answer.

## 7. Limitations of the Current Evaluation

Our test set contains only three queries.

This is useful for learning, but too small to establish robust retrieval quality.

We also define correctness at the source-file level. In a larger evaluation, we might label individual chunks as relevant or irrelevant.

Useful additional metrics include:

- **Precision@K:** The fraction of retrieved top-K results that are relevant.
- **Recall@K:** The fraction of all known relevant items that appear in the top K.
- **Mean Reciprocal Rank (MRR):** Rewards systems that rank the first relevant result highly.
- **nDCG:** Evaluates ranking quality when relevance has multiple levels.

The correct metric depends on the application and how relevance is defined.

## 8. Key Takeaways

- Retrieval should be evaluated using representative queries.
- Hit@K checks whether an expected relevant source appears among the top K results.
- A high Hit@K does not guarantee that every result is relevant.
- Small test sets provide only limited evidence.
- Retrieval evaluation is separate from evaluating the final generated answer.
- Evaluation helps us make informed decisions about chunking, embeddings, and retrieval settings.



**1.Semantic search:** Which chunks are most similar to this query?
**2.Hit@K:** Did at least one expected/relevant result appear among the top K?
**3.Expected source:** The ground-truth filename that we supplied in our test case.