# Lesson 2: Loading Text Documents with Python

## 1. Learning objectives

- Load multiple text files from a directory.
- Use Python's `pathlib` module to work with paths.
- Read text using UTF-8 encoding.
- Create document records with metadata.
- Calculate character and word counts.
- Validate that all expected documents were loaded.

## 2. Why do we load documents?

A real RAG system usually works with external knowledge sources, such as troubleshooting guides, internal documentation, manuals, and support articles.

The first step is to read these sources into a format that Python can process.

Our initial example uses three text files. Later, we can adapt the same workflow to other formats and document loaders.

## 3. Understanding pathlib

`pathlib` is a Python standard-library module for working with filesystem paths.

```python
from pathlib import Path

folder = Path("data") / "sample_docs"
```

The `/` operator joins path components. It does not divide numbers in this context.

Using `pathlib` is generally more portable than manually concatenating path strings.

## 4. Finding files

```python
file_paths = sorted(folder.glob("*.txt"))
```

`glob("*.txt")` finds text files directly inside the selected directory.

`sorted()` arranges the paths in a predictable order.

If files are inside nested folders, `glob()` does not discover them recursively. Use `rglob("*.txt")` if recursive discovery is required.

## 5. Reading file contents

```python
text = file_path.read_text(encoding="utf-8")
```

This reads the entire file into a Python string.

For very large files, reading everything at once might consume substantial memory. Streaming or processing files in smaller parts may be more appropriate.

## 6. Creating document records

```python
document = {
    "id": file_path.stem,
    "text": text.strip(),
    "source": file_path.name,
    "source_path": str(file_path),
}
```

- `id` identifies the record.
- `text` contains the document content.
- `source` preserves the filename.
- `source_path` records the path used by the script.

The example uses a simple filename-based ID. In a larger system, filenames might not be unique, so IDs may need a more robust strategy.

## 7. Why strip the text?

`strip()` removes whitespace from the beginning and end of a string.

It does not remove all internal spaces or reformat the entire document.

Preserving meaningful whitespace matters for content such as source code, tables, and structured documents.

## 8. Text statistics

```python
character_count = len(text)
word_count = len(text.split())
```

`len(text)` counts characters. `len(text.split())` counts whitespace-separated pieces.

The latter is a simple word-count approximation, not a tokenizer count. Language-model token counts are calculated differently.

## 9. Why validate the loaded documents?

```python
assert len(documents) == 3
assert all(document["text"] for document in documents)
```

The first assertion checks the expected number of documents. The second checks that each record contains non-empty text.

Assertions are useful for this learning exercise. Production pipelines should generally use explicit error handling and clear validation messages.

## 10. Connection to RAG

The pipeline now looks like this:

1. Locate the source files.
2. Read their contents.
3. Create document records.
4. Preserve source metadata.
5. Validate the collection.
6. Prepare the documents for future chunking and embedding.

We have not yet generated embeddings or performed semantic search.

## 11. Practice questions

1. What does `Path(__file__).parent` represent?
2. What is the difference between `file_path.stem` and `file_path.name`?
3. What does `glob("*.txt")` do?
4. Why do we specify UTF-8 encoding?
5. Why preserve source information in each document?
6. What would happen if the directory contained four text files instead of three with the current assertion?

## 12. Completion checklist

- [ ] Create the three sample text files.
- [ ] Run `02_load_text_documents.py`.
- [ ] Understand the document dictionary.
- [ ] Explain the role of `pathlib`.
- [ ] Answer the six practice questions.