
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "sample_docs"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3


# --------------------------------------------------
# 2. Load documents and prepare chunks
# --------------------------------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=80,
    chunk_overlap=20,
    separators=["\n\n", "\n", " ", ""],
)

documents = []

for file_path in sorted(DATA_DIR.glob("*.txt")):
    text = file_path.read_text(encoding="utf-8")
    text = " ".join(text.split())

    documents.append({
        "source": file_path.name,
        "text": text,
    })

chunks = []

for document in documents:
    split_texts = splitter.split_text(document["text"])
    source_stem = Path(document["source"]).stem

    for index, chunk_text in enumerate(split_texts):
        chunks.append({
            "id": f"{source_stem}_chunk_{index:03d}",
            "source": document["source"],
            "text": chunk_text,
        })

print(f"Documents loaded: {len(documents)}")
print(f"Chunks created: {len(chunks)}")


# --------------------------------------------------
# 3. Generate document embeddings
# --------------------------------------------------

model = SentenceTransformer(MODEL_NAME)

chunk_texts = [chunk["text"] for chunk in chunks]

document_embeddings = model.encode(
    chunk_texts,
    convert_to_numpy=True,
    normalize_embeddings=True,
)

print(f"Embedding matrix shape: {document_embeddings.shape}")


# --------------------------------------------------
# 4. Define evaluation test cases
# --------------------------------------------------
# These relevance labels are created manually.
# They are the ground truth, not model predictions.

test_cases = [
    {
        "query": "How can I regain access to my account?",
        "relevant_chunk_ids": [
            "account_chunk_000",
            "account_chunk_001",
        ],
    },
    {
        "query": "What should I do if the application crashes?",
        "relevant_chunk_ids": [
            "technical_chunk_000",
            "technical_chunk_001",
        ],
    },
    {
        "query": "How do I handle duplicate payments?",
        "relevant_chunk_ids": [
            "billing_chunk_000",
            "billing_chunk_001",
        ],
    },
]


# --------------------------------------------------
# 5. Retrieve the top K chunks
# --------------------------------------------------

def semantic_search(query, top_k=TOP_K):
    query_embedding = model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # Normalized embeddings make dot product equal
    # to cosine similarity.
    scores = document_embeddings @ query_embedding

    ranked_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in ranked_indices:
        result = chunks[int(index)].copy()
        result["score"] = float(scores[index])
        results.append(result)

    return results


# --------------------------------------------------
# 6. Calculate retrieval evaluation metrics
# --------------------------------------------------

def evaluate_retrieval(results, relevant_chunk_ids, k):
    relevant_set = set(relevant_chunk_ids)

    # Only evaluate the first K retrieved results.
    top_results = results[:k]
    retrieved_ids = [result["id"] for result in top_results]

    relevant_retrieved = [
        chunk_id
        for chunk_id in retrieved_ids
        if chunk_id in relevant_set
    ]

    # Hit@K: Was at least one relevant chunk retrieved?
    hit_at_k = int(len(relevant_retrieved) > 0)

    # Precision@K: What fraction of top K results are relevant?
    precision_at_k = len(relevant_retrieved) / k

    # Recall@K: What fraction of all known relevant chunks
    # were retrieved?
    recall_at_k = len(relevant_retrieved) / len(relevant_set)

    # Reciprocal rank: 1 / rank of the first relevant result.
    reciprocal_rank = 0.0

    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_set:
            reciprocal_rank = 1.0 / rank
            break

    return {
        "hit_at_k": hit_at_k,
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
        "reciprocal_rank": reciprocal_rank,
        "relevant_retrieved": len(relevant_retrieved),
        "total_relevant": len(relevant_set),
    }


# --------------------------------------------------
# 7. Run evaluation across all test queries
# --------------------------------------------------

all_metrics = []

print("\nMEDIUM-LEVEL RETRIEVAL EVALUATION")
print("=" * 65)

for case in test_cases:
    query = case["query"]
    relevant_ids = case["relevant_chunk_ids"]

    results = semantic_search(query, top_k=TOP_K)

    metrics = evaluate_retrieval(
        results,
        relevant_ids,
        TOP_K,
    )
    all_metrics.append(metrics)

    print(f"\nQuery: {query}")
    print(f"Expected relevant chunks: {relevant_ids}")

    print("\nRetrieved results:")

    for rank, result in enumerate(results, start=1):
        is_relevant = result["id"] in relevant_ids
        label = "RELEVANT" if is_relevant else "NOT RELEVANT"

        print(
            f"  {rank}. {result['id']} | "
            f"{result['source']} | "
            f"score={result['score']:.4f} | {label}"
        )
        print(f"     Text: {result['text']}")

    print("\nMetrics for this query:")
    print(f"  Hit@{TOP_K}:       {metrics['hit_at_k']}")
    print(f"  Precision@{TOP_K}:  {metrics['precision_at_k']:.4f}")
    print(f"  Recall@{TOP_K}:     {metrics['recall_at_k']:.4f}")
    print(f"  Reciprocal Rank:    {metrics['reciprocal_rank']:.4f}")
    print(
        f"  Relevant retrieved: "
        f"{metrics['relevant_retrieved']}/"
        f"{metrics['total_relevant']}"
    )


# --------------------------------------------------
# 8. Calculate average metrics
# --------------------------------------------------

number_of_queries = len(all_metrics)

average_hit = sum(
    metric["hit_at_k"] for metric in all_metrics
) / number_of_queries

average_precision = sum(
    metric["precision_at_k"] for metric in all_metrics
) / number_of_queries

average_recall = sum(
    metric["recall_at_k"] for metric in all_metrics
) / number_of_queries

mean_reciprocal_rank = sum(
    metric["reciprocal_rank"] for metric in all_metrics
) / number_of_queries


# --------------------------------------------------
# 9. Print evaluation summary
# --------------------------------------------------

print("\n\nOVERALL EVALUATION SUMMARY")
print("=" * 65)
print(f"Queries evaluated: {number_of_queries}")
print(f"Hit@{TOP_K}:       {average_hit:.4f}")
print(f"Precision@{TOP_K}:  {average_precision:.4f}")
print(f"Recall@{TOP_K}:     {average_recall:.4f}")
print(f"MRR:                {mean_reciprocal_rank:.4f}")
print("=" * 65)
