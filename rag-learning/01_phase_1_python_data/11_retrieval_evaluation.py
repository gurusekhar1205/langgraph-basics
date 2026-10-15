
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


DATA_DIR = Path(__file__).parent / "data" / "sample_docs"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 80
CHUNK_OVERLAP = 20
TOP_K = 3


# 1. Load model and splitter
model = SentenceTransformer(MODEL_NAME)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    separators=["\n\n", "\n", " ", ""],
)


# 2. Load documents and create chunks
chunks = []

for file_path in sorted(DATA_DIR.glob("*.txt")):
    text = " ".join(
        file_path.read_text(encoding="utf-8").split()
    )

    if not text:
        continue

    document_id = file_path.stem

    for index, chunk_text in enumerate(splitter.split_text(text)):
        chunks.append(
            {
                "chunk_id": f"{document_id}_chunk_{index:03d}",
                "source": file_path.name,
                "text": chunk_text,
            }
        )


# 3. Embed all chunks once
document_texts = [chunk["text"] for chunk in chunks]

document_embeddings = model.encode(
    document_texts,
    normalize_embeddings=True,
    show_progress_bar=True,
)


# 4. Define the search function
def semantic_search(query, top_k=TOP_K):
    if not query.strip():
        raise ValueError("Query cannot be empty.")

    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
    )

    scores = document_embeddings @ query_embedding
    ranked_indices = np.argsort(scores)[::-1]

    results = []

    for index in ranked_indices[:min(top_k, len(chunks))]:
        chunk = chunks[int(index)]

        results.append(
            {
                "chunk_id": chunk["chunk_id"],
                "source": chunk["source"],
                "text": chunk["text"],
                "score": float(scores[index]),
            }
        )

    return results


# 5. Define a small evaluation dataset
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


# 6. Evaluate retrieval using Hit@K
hits = 0

print("\nRetrieval Evaluation")
print("=" * 60)

for test_case in test_cases:
    query = test_case["query"]
    expected_source = test_case["expected_source"]

    results = semantic_search(query, top_k=TOP_K)

    retrieved_sources = [
        result["source"] for result in results
    ]

    hit = expected_source in retrieved_sources

    if hit:
        hits += 1

    print("\nQuery:", query)
    print("Expected source:", expected_source)
    print("Retrieved sources:", retrieved_sources)
    print("Hit@K:", hit)

    for rank, result in enumerate(results, start=1):
        print(
            f"  {rank}. score={result['score']:.4f} "
            f"| {result['source']} "
            f"| {result['text']}"
        )


# 7. Calculate overall Hit@K
hit_rate = hits / len(test_cases) if test_cases else 0.0

print("\nEvaluation Summary")
print("=" * 60)
print("Queries tested:", len(test_cases))
print("Successful hits:", hits)
print(f"Hit@{TOP_K}: {hit_rate:.2%}")
