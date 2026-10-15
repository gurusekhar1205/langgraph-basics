
# Lesson 12: Medium-Level RAG Retrieval Evaluation

## 1. Objective

In Lesson 11, we evaluated retrieval using Hit@K and expected source filenames.

In this lesson, we extend evaluation with:

- Hit@K
- Precision@K
- Recall@K
- Reciprocal Rank
- Mean Reciprocal Rank (MRR)
- Chunk-level relevance labels

The objective is to evaluate retrieval quality, not merely whether a program executes successfully.

## 2. What is ground truth?

Ground truth is the expected correct information used to evaluate a system.

For retrieval, we manually specify which chunk IDs are relevant to each query.

Example:

Query:
"How can I regain access to my account?"

Relevant chunk IDs:
- account_chunk_000
- account_chunk_001

These labels are our reference for evaluation.

The embedding model does not create these labels. We define them based on the document content and the question.

Important:
Relevance labels are query-specific. A chunk can be relevant to one query and irrelevant to another.

## 3. Hit@K

Hit@K checks whether at least one relevant chunk appears among the top K results.

Formula:

Hit@K = 1 if at least one relevant chunk is retrieved.
Hit@K = 0 otherwise.

If the first three results are:
1. Relevant
2. Irrelevant
3. Irrelevant

Hit@3 = 1.

Limitation:
Hit@K does not measure how many results are relevant.

## 4. Precision@K

Precision@K measures the fraction of the top K results that are relevant.

Formula:

Precision@K =
Relevant chunks among top K / K

Example:

Top 3 results:
1. Relevant
2. Irrelevant
3. Relevant

Precision@3 = 2 / 3 = 0.6667.

Interpretation:
Approximately 66.67% of the top three retrieved chunks are relevant.

Higher precision generally means less irrelevant context is retrieved.

## 5. Recall@K

Recall@K measures the fraction of all known relevant chunks that appear among the top K results.

Formula:

Recall@K =
Relevant chunks retrieved in top K / Total relevant chunks

Example:

There are four known relevant chunks.
The retriever finds two of them in the top three.

Recall@3 = 2 / 4 = 0.5.

Interpretation:
The retriever found 50% of the known relevant chunks.

Important:
Recall requires knowing how many chunks are relevant in the evaluation dataset.
The relevant_chunk_ids list supplies that information for each test query.

## 6. Reciprocal Rank

Reciprocal Rank measures the position of the first relevant result.

Formula:

RR = 1 / Rank of first relevant result

Examples:

First relevant result at rank 1:
RR = 1 / 1 = 1.0

First relevant result at rank 2:
RR = 1 / 2 = 0.5

First relevant result at rank 3:
RR = 1 / 3 = 0.3333

No relevant result:
RR = 0.

A higher reciprocal rank means the first relevant chunk appears earlier.

## 7. Mean Reciprocal Rank (MRR)

MRR is the average reciprocal rank across all evaluation queries.

Formula:

MRR = Sum of reciprocal ranks / Number of queries

Example:

Query 1: RR = 1.0
Query 2: RR = 0.5
Query 3: RR = 0.3333

MRR = (1.0 + 0.5 + 0.3333) / 3
MRR = approximately 0.6111.

MRR emphasizes the first relevant result.
It does not measure all relevant results.

## 8. How semantic search works in this lesson

1. Read the text documents.
2. Split each document into chunks.
3. Attach a chunk ID and source filename to each chunk.
4. Generate an embedding for every chunk.
5. Generate an embedding for the user query.
6. Calculate cosine similarity using normalized vectors.
7. Sort the similarity scores in descending order.
8. Select the top K chunks.
9. Compare their IDs with the manually labelled relevant chunk IDs.
10. Calculate evaluation metrics.

The embedding model retrieves candidate chunks.
The evaluation labels determine which retrieved chunks count as relevant.

## 9. Why chunk IDs matter

Checking filenames is useful for initial testing, but it is not sufficiently precise for every question.

For example, two chunks can come from account.txt:
- One discusses password resets.
- Another discusses a different account-related issue.

The first might be relevant to a password-reset query while the second is not.

Chunk-level evaluation allows us to distinguish them.

## 10. Understanding the Python implementation

### Building the corpus

The script reads the sample text files and creates chunk records.

Each record contains:
- id: Unique chunk identifier
- source: Original filename
- text: Chunk content

### Creating embeddings

SentenceTransformer converts each chunk into a 384-dimensional vector using all-MiniLM-L6-v2.

The vectors are normalized.

### Retrieving candidates

The query is embedded and compared with all chunk embeddings.

The highest similarity scores are ranked first.

### Applying ground truth

The evaluation function compares retrieved chunk IDs with relevant_chunk_ids.

This comparison is independent of the similarity score.

### Calculating metrics

The function computes Hit@K, Precision@K, Recall@K, and Reciprocal Rank for each query.

The script then averages the metrics across all test cases.

## 11. How to run the program

From this folder:

rag-learning/01_phase_1_python_data/

Run:

python 12_retrieval_evaluation_metrics.py

The program prints:
- Number of documents and chunks
- Retrieved chunks and similarity scores
- Relevance labels
- Metrics for each query
- Overall average metrics

The exact values depend on the retrieved ranking and the relevance labels.

## 12. Limitations

1. The evaluation dataset contains only three queries.
2. The relevance labels are manually defined.
3. Labels may be incomplete or subjective.
4. The evaluation tests retrieval, not final LLM answer quality.
5. Precision@K uses K as the denominator.
6. Recall depends on the completeness of the known relevant chunk IDs.
7. MRR focuses on the first relevant result.

A high metric on a tiny dataset does not prove that a production RAG system is reliable.

## 13. Exercises

1. Change TOP_K from 3 to 2. Observe how the metrics change.
2. Add a query that has no relevant chunks in the top three.
3. Add another relevant chunk ID to a test case and observe Recall@K.
4. Compare Hit@K and Precision@K for the same query.
5. Explain why MRR can be high even when precision is low.

## 14. Key takeaways

- Hit@K checks whether at least one relevant result was retrieved.
- Precision@K measures the relevance of retrieved results.
- Recall@K measures how many known relevant results were found.
- Reciprocal Rank rewards an early first relevant result.
- MRR averages reciprocal rank across queries.
- Ground truth must be defined independently of retrieval predictions.
- Chunk-level relevance is more precise than checking only filenames.
