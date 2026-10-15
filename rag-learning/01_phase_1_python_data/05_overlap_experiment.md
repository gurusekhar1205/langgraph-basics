# Experiment 05: Understanding Chunk Overlap in RAG

## 1. Objective

The objective of this experiment is to understand how **chunk overlap** works when splitting documents into smaller pieces for a Retrieval-Augmented Generation (RAG) system.

We will learn:

- Why documents are divided into chunks.
- What `chunk_size` and `chunk_overlap` mean.
- How overlapping chunks preserve context.
- Why the actual overlap may differ from the requested overlap.
- How to inspect and evaluate the output of a text splitter.

---

## 2. Why Do We Need Chunking?

In a RAG system, we usually retrieve relevant information from documents and provide it to a language model to answer a user's question.

Suppose a document contains 10,000 characters. Passing the entire document to the model for every question may be inefficient.

Instead, we divide the document into smaller pieces called **chunks**.

For example:

**Original document:**

```text
Users can reset a forgotten password from the login page.
Contact support if the reset link expires.
Billing issues include duplicate payments and incorrect invoices.
```

**After chunking:**

```text
Chunk 1:
Users can reset a forgotten password from the login page.

Chunk 2:
Contact support if the reset link expires.

Chunk 3:
Billing issues include duplicate payments and incorrect invoices.
```

A retriever can search these chunks and select the ones most relevant to a user's question.

However, chunking introduces a potential problem: **important context may be located near the boundary between two chunks.**

Chunk overlap helps reduce this problem.

---

## 3. What Is Chunk Size?

`chunk_size` specifies the target maximum length of each chunk, according to the configured length function.

In our experiment:

```python
chunk_size=50
```

We use `RecursiveCharacterTextSplitter` with its default length function, which counts characters.

Therefore, the splitter aims to create chunks containing no more than 50 characters.

For example:

```text
This is the first sentence. This is the second
```

This chunk contains 46 characters, so it fits within the configured size.

**Important:** A chunk size of 50 does not mean 50 words or 50 tokens. In this experiment, it means 50 characters.

---

## 4. What Is Chunk Overlap?

`chunk_overlap` specifies the target amount of text that should be shared between consecutive chunks.

Our experiment uses:

```python
chunk_overlap=10
```

The splitter attempts to retain approximately 10 characters of text from one chunk in the next chunk.

Consider this simplified example:

```text
Chunk 1: ABCDEFGHIJKLMNOP
Chunk 2:       MNOPQRSTUVWX
```

The text `MNOP` is repeated in both chunks.

That repeated text is called **overlap**.

### Why is overlap useful?

Imagine that a document contains this information:

```text
The application reports an authentication error.
To resolve this issue, refresh the access token.
```

If the document is divided at an unfortunate boundary, the explanation and the resolution might end up in separate chunks.

Overlap can preserve some surrounding context near the boundary, making it easier for retrieval to find useful information.

However, overlap does not guarantee that the complete explanation and resolution will be stored together.

---

## 5. Python Code Used in This Experiment

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=10,
)

text = (
    "This is the first sentence. "
    "This is the second sentence. "
    "This is the third sentence. "
    "This is the fourth sentence."
)

chunks = splitter.split_text(text)

print("Original text length:", len(text))
print("Number of chunks:", len(chunks))
print()

for i, chunk in enumerate(chunks):
    print("Chunk index:", i)
    print("Chunk text:", repr(chunk))
    print("Character count:", len(chunk))
    print("-" * 50)
```

### Explanation of the important statements

**1. Create the splitter**

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=10,
)
```

This configures the splitter to target chunks of up to 50 characters, with a requested overlap of 10 characters.

**2. Split the text**

```python
chunks = splitter.split_text(text)
```

This returns a list of strings. Each string represents one chunk.

**3. Enumerate the chunks**

```python
for i, chunk in enumerate(chunks):
```

`enumerate()` gives us both the index and the text of each chunk.

**4. Display the character count**

```python
len(chunk)
```

This tells us the number of characters in the current chunk.

**5. Display the representation**

```python
repr(chunk)
```

`repr()` displays a readable representation of the string, helping us inspect spaces and other characters.

---

## 6. Actual Output

The experiment produced the following output:

```text
Original text length: 113
Number of chunks: 3

Chunk index: 0
Chunk text: 'This is the first sentence. This is the second'
Character count: 46

Chunk index: 1
Chunk text: 'second sentence. This is the third sentence. This'
Character count: 49

Chunk index: 2
Chunk text: 'This is the fourth sentence.'
Character count: 28
```

Let's understand each result.

### Chunk 0

```text
This is the first sentence. This is the second
```

- Length: 46 characters.
- It contains the first sentence and the beginning of the second sentence.
- It is shorter than the configured limit of 50 characters.

### Chunk 1

```text
second sentence. This is the third sentence. This
```

- Length: 49 characters.
- It continues the second sentence and includes the third sentence.
- It also contains text repeated from Chunk 0.

### Chunk 2

```text
This is the fourth sentence.
```

- Length: 28 characters.
- It contains the fourth sentence.
- It is shorter than the configured limit because the remaining text is insufficient to form another full-sized chunk.

All three chunks are within the configured 50-character limit.

---

## 7. How Much Text Actually Overlaps?

This is the most important observation from the experiment.

The configured overlap is 10 characters, but the actual overlap is determined by comparing consecutive chunks.

### Between Chunk 0 and Chunk 1

Chunk 0 ends with:

```text
...This is the second
```

Chunk 1 begins with:

```text
second sentence...
```

The exact shared text is:

```text
second
```

There are **6 shared characters** in `second`, not 10.

### Between Chunk 1 and Chunk 2

Chunk 1 ends with:

```text
...sentence. This
```

Chunk 2 begins with:

```text
This is the fourth sentence.
```

The exact shared text is:

```text
This
```

There are **4 shared characters** in `This`, not 10.

### Why is the overlap not exactly 10 characters?

`RecursiveCharacterTextSplitter` does not simply cut the text at fixed character positions and copy exactly 10 characters each time.

It first tries to split text using meaningful separators, such as paragraph breaks, line breaks, and spaces. It then merges smaller pieces into chunks while considering the configured size and overlap.

The resulting overlap depends on the boundaries of those pieces and the merge process.

Therefore:

**`chunk_overlap=10` is a target, not a guarantee that every neighboring pair of chunks will share exactly 10 characters.**

---

## 8. How Does Recursive Character Splitting Work?

`RecursiveCharacterTextSplitter` tries progressively smaller separators to divide text.

Its default separator preference is approximately:

1. `"\n\n"` — paragraph breaks.
2. `"\n"` — line breaks.
3. `" "` — spaces between words.
4. `""` — individual characters when necessary.

The splitter prefers larger, more meaningful boundaries before resorting to smaller ones.

For example, it may prefer splitting at a paragraph boundary rather than cutting a word in half.

However, it does not guarantee that every chunk contains complete sentences. A sentence may still be split if the configured size requires it.

### Recursive splitting versus fixed character slicing

| Fixed character slicing | Recursive character splitting |
|---|---|
| Cuts text at predetermined character positions. | Tries preferred text separators first. |
| Can easily split words. | Usually preserves words when possible. |
| Does not naturally prioritize paragraphs or lines. | Can prioritize paragraphs, lines, and spaces. |
| Simple and predictable by position. | More flexible for ordinary documents. |

For many document-based RAG projects, recursive splitting is a useful starting point.

---

## 9. Is Overlap Always Necessary?

No. Overlap is a design choice.

### Without overlap

```python
RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=0,
)
```

Advantages:

- Less repeated text.
- Fewer total characters stored in the vector database.
- Potentially lower embedding and storage costs.

Disadvantages:

- Context near chunk boundaries may be harder to retrieve together.
- Related information may be separated across chunks.

### With overlap

```python
RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)
```

Advantages:

- Repeats some context near boundaries.
- Can help retrieval when relevant information crosses a chunk boundary.

Disadvantages:

- Produces repeated text.
- Can increase the number of chunks, embedding costs, and storage requirements.
- May retrieve redundant information.

A larger overlap is not automatically better. The best configuration depends on the documents, queries, retrieval strategy, and evaluation results.

---

## 10. Important Limitations to Remember

1. **Overlap does not merge chunks.** Each chunk remains a separate string.
2. **Overlap does not guarantee complete context.** A long explanation can still span multiple chunks.
3. **Overlap is not always exact.** The configured value is a target.
4. **Chunk size is not automatically a token count.** Our experiment counts characters.
5. **Recursive splitting does not guarantee complete sentences.** It prefers configured separators but can split sentences.
6. **Good-looking chunks do not automatically produce good retrieval.** We must test whether relevant queries retrieve the correct information.

---

## 11. How This Fits into a RAG Pipeline

Our current learning pipeline is:

```text
Source documents
      |
      v
Load documents
      |
      v
Clean text
      |
      v
Split text into chunks
      |
      v
Attach metadata
      |
      v
Generate embeddings
      |
      v
Store vectors in a vector database
      |
      v
Retrieve relevant chunks for a query
      |
      v
Give retrieved context to the language model
      |
      v
Generate an answer
```

Chunking and overlap happen before embedding generation.

Each chunk will eventually receive an embedding, a numerical representation of its content. A retriever can then use these representations to find chunks relevant to a user's question.

---

## 12. Key Takeaways

- Chunking divides long documents into smaller pieces.
- `chunk_size` controls the target maximum length.
- `chunk_overlap` requests repeated text between neighboring chunks.
- Overlap can preserve context near chunk boundaries.
- The actual overlap may be smaller or otherwise differ from the requested value.
- Recursive splitting prefers meaningful separators but does not guarantee complete sentences.
- Overlap introduces a trade-off between context preservation and redundant storage.
- Retrieval quality must be evaluated using relevant queries, not just by looking at the chunks.

---

## 13. Practice Questions

Try answering these without looking at the explanations above.

1. Why do we split documents before generating embeddings?
2. What is the difference between `chunk_size=100` and `chunk_overlap=20`?
3. Why did the first two chunks share the word `second`?
4. Why did the actual overlap differ from the requested 10 characters?
5. Does setting `chunk_overlap=0` mean RAG cannot retrieve useful information?
6. Why might a very large overlap increase cost?
7. Does recursive splitting guarantee that every chunk contains complete sentences?
8. Why should we evaluate retrieval quality instead of choosing chunk settings based only on appearance?

**Next lesson:** Learn how to inspect chunk metadata and prepare a consistent list of document chunks for embedding generation.