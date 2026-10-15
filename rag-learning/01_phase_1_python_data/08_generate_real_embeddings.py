
from sentence_transformers import SentenceTransformer
import numpy as np

# 1. Load a pretrained embedding model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

print("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)

# 2. Prepare sample sentences
sentences = [
    "How can I reset my password?",
    "How do I recover my forgotten password?",
    "What is the weather today?",
    "I cannot log in because I forgot my credentials.",
]

# 3. Generate real embeddings
embeddings = model.encode(sentences)

# 4. Inspect the output
print("\nEmbedding results")
print("=" * 60)

print("Number of sentences:", len(sentences))
print("Embedding matrix shape:", embeddings.shape)
print("Vector dimensions:", embeddings.shape[1])

for index, sentence in enumerate(sentences):
    print(f"\nSentence {index}: {sentence}")
    print("First 5 vector values:", embeddings[index][:5])
    print("Vector length:", len(embeddings[index]))

# 5. Compare sentence similarity using cosine similarity
normalized_embeddings = embeddings / np.linalg.norm(
    embeddings, axis=1, keepdims=True
)

similarity_matrix = normalized_embeddings @ normalized_embeddings.T

print("\n\nCosine similarity matrix")
print("=" * 60)

for i, sentence in enumerate(sentences):
    print(f"\nSentence {i}: {sentence}")

    for j, other_sentence in enumerate(sentences):
        print(
            f"  vs Sentence {j}: "
            f"{similarity_matrix[i, j]:.4f}"
        )
